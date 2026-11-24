"""Optional fixed global batches and resumable warmup/cosine learning rates.

Legacy token-budget training is unchanged unless global_batch_size is supplied.
The schedule is a pure function of committed optimizer_steps, so prefetching and
checkpoint reloads cannot advance it accidentally.
"""
import math

from valen.data.prefetch import PackPrefetcher


def validate_schedule(config):
    batch = config.get("global_batch_size")
    if batch is not None and (type(batch) is not int or batch < 1):
        raise ValueError("global_batch_size must be a positive integer")
    kind = config.get("lr_scheduler", "constant")
    if kind not in {"constant", "cosine"}:
        raise ValueError("lr_scheduler must be constant or cosine")
    warmup = config.get("warmup_ratio", 0.0)
    if isinstance(warmup, bool) or not isinstance(warmup, (int, float)) or not math.isfinite(warmup) or not 0 <= warmup < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if kind == "constant" and warmup:
        raise ValueError("warmup_ratio requires lr_scheduler=cosine")
    if kind == "cosine" and batch is None:
        raise ValueError("Cosine scheduling requires global_batch_size to determine the step count")
    if batch is not None:
        if config.get("method", "sft") != "sft" or config.get("sft_iterations", 1) != 1:
            raise ValueError("Fixed global batches currently require single-iteration SFT")
        if type(config.get("epochs", 1)) is not int or config.get("epochs", 1) < 1:
            raise ValueError("Fixed global batches require positive integer epochs")
        if "max_steps" in config and (type(config["max_steps"]) is not int or config["max_steps"] < 1):
            raise ValueError("max_steps must be a positive integer")
    return config


def per_rank_batch(config, world_size):
    batch = config["global_batch_size"]
    if batch % world_size:
        raise ValueError("global_batch_size must be divisible by world_size")
    return batch // world_size


class FixedBatchPrefetcher(PackPrefetcher):
    def __init__(self, records, compiler, backend, config, *, world_size):
        super().__init__(records, compiler, backend, config)
        self.batch_size = per_rank_batch(config, world_size)

    def _prepare(self, order, cursor, rng_state):
        end = min(len(order), cursor + self.batch_size)
        # A view avoids copying an ever-growing prefix on every optimizer step.
        class Prefix:
            def __len__(self):
                return end

            def __getitem__(self, index):
                return order[index]

        result = super()._prepare(Prefix(), cursor, rng_state)
        # Never silently use smaller batches because of a token budget or missing
        # labels: that would invalidate the planned cosine schedule.
        if result.cursor != end or len(result.states) != end - cursor:
            raise ValueError("Fixed global batch was truncated: increase tokens_per_step and ensure all records are labeled")
        return result


def make_prefetcher(records, compiler, backend, config, world_size):
    if config.get("global_batch_size") is not None:
        return FixedBatchPrefetcher(records, compiler, backend, config, world_size=world_size)
    return PackPrefetcher(records, compiler, backend, config)


class LearningRateSchedule:
    def __init__(self, config, optimizer, record_count, world_size):
        validate_schedule(config)
        self.kind = config.get("lr_scheduler", "constant")
        self.optimizer = optimizer
        self.steps = self.warmup = None
        if config.get("global_batch_size") is not None:
            per_rank_batch(config, world_size)
            self.steps = min(math.ceil(record_count / config["global_batch_size"]) * config.get("epochs", 1),
                             config.get("max_steps", 2**63 - 1))
            if self.steps < 1:
                raise ValueError("Cannot schedule an empty dataset")
            self.warmup = math.ceil(self.steps * config.get("warmup_ratio", 0.0))
        if self.kind == "cosine":
            # initial_lr persists in optimizer checkpoints. A task initialization
            # creates a fresh optimizer, resetting the schedule for the new task.
            self.base_lrs = [group.setdefault("initial_lr", group["lr"]) for group in optimizer.param_groups]

    def factor(self, step):
        if self.kind == "constant":
            return 1.0
        if step < self.warmup:
            return step / max(1, self.warmup)
        progress = min(1.0, max(0.0, (step - self.warmup) / max(1, self.steps - self.warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    def apply(self, step):
        if self.kind == "cosine":
            factor = self.factor(step)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * factor

    def metadata(self):
        return {"kind": self.kind, "planned_steps": self.steps, "warmup_steps": self.warmup}
