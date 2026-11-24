"""Shared SFT/RLCD loop: token budgets, distributed updates and exact resume."""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path
import torch
from valen.training.checkpoint import capture_rank_state, file_sha256, load_checkpoint, save_checkpoint
from valen.training.distributed import initialize as initialize_distributed, close, epoch_shard, broadcast_trainable, synchronize_gradients
from .sft import SFTObjective
from .batching import training_batches
from .rlcd import RLCDObjective, validate_options
from valen.modeling.factory import build_model, build_compiler, get_backend, normalize_model_config, optimizer_groups
from valen.data.schema import read_jsonl
from .schedule import LearningRateSchedule, make_prefetcher, validate_schedule
from .diagnostics import PhaseTrace, trace_options
from valen.modeling.qwen.attention import ATTENTION_IMPLEMENTATIONS, with_attention_implementation
from tqdm import tqdm


class StepTimer:
    """分段诊断计时：CUDA 边界等待本卡完成，统计 CPU 调度及 GPU 执行墙钟时间。

    不插入跨 rank barrier；通信阶段包含等待其他 rank 的时间。
    同步会影响计算重叠，因此这是诊断耗时，不是无干扰的性能 benchmark。
    """

    def __init__(self, device, trace=None):
        self.device = torch.device(device)
        self.trace = trace if trace is not None else PhaseTrace(0, 0)
        self.seconds = dict.fromkeys(("data", "counts", "prepare", "zero_grad", "forward",
                                     "backward", "metrics", "sync", "clip", "optimizer"), 0.0)
        with self.trace.phase("step_start"):
            self.synchronize()
        self.started = time.monotonic()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        # Log before CUDA synchronization so a blocked synchronize is visible too.
        with self.trace.phase(name):
            self.trace.emit(name, "wait_before")
            self.synchronize()
            started = time.monotonic()
            self.trace.emit(name, "body")
            try:
                yield
            finally:
                self.trace.emit(name, "wait_after")
                self.synchronize()
                self.seconds[name] += time.monotonic() - started

    def snapshot(self):
        self.synchronize()
        return dict(self.seconds, total=time.monotonic() - self.started)


def timing_postfix(step, seconds, training_units):
    # 秒；X 汇总统计通信、rollout、清梯度、指标提取和裁剪。
    other = sum(seconds[name] for name in ("counts", "prepare", "zero_grad", "metrics", "clip"))
    fields = {"D": seconds["data"], "F": seconds["forward"], "B": seconds["backward"],
              "S": seconds["sync"], "O": seconds["optimizer"], "X": other, "T": seconds["total"]}
    return f"step={step} u={training_units} " + " ".join(f"{key}={value:.3f}s" for key, value in fields.items())


def normalize_config(config):
    """Normalize architecture, objective settings and disabled legacy fields."""
    config = normalize_model_config(config)
    if config.get("kd_weight", 0) != 0 or config.get("teacher_checkpoint"):
        raise ValueError("Teacher distillation was removed; remove teacher_checkpoint and nonzero kd_weight")
    for key in ("kd_weight", "kd_temperature", "teacher_checkpoint"):
        config.pop(key, None)
    method = config.setdefault("method", "sft")
    for name in ("rps_weight", "brier_weight"):
        value = config.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if method not in {"sft", "rlcd"}:
        raise ValueError(f"Unknown training method: {method}")
    if method == "rlcd":
        config["rlcd"] = validate_options(config.get("rlcd", {}))
        if config.get("rps_weight", 0):
            raise ValueError("RLCD uses Brier calibration; rps_weight must be zero")
    elif config.get("rlcd"):
        raise ValueError("Set method=rlcd to use RLCD options")
    if config.setdefault("loss_reduction", "state_mean") not in ("state_mean", "question_mean"):
        raise ValueError("loss_reduction must be state_mean or question_mean")
    for name, default in (("tokens_per_step", 16384), ("microbatch_size", 1), ("microbatch_max_tokens", 32768)):
        value = config.setdefault(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    shared_rl = (method == "rlcd" and config.get("qwen_execution") == "shared_state"
                 and not config.get("cache_frozen_features", False))
    if config["microbatch_size"] > 1 and (config["architecture"] != "qwen" or not (method == "sft" or shared_rl)):
        raise ValueError("Parallel microbatches require Qwen SFT or shared_state RLCD without feature caching")
    if type(config.setdefault("data_prefetch", False)) is not bool:
        raise ValueError("data_prefetch must be a boolean")
    if config["data_prefetch"] and (config["architecture"] != "qwen" or not (method == "sft" or shared_rl)):
        raise ValueError("Data prefetch requires Qwen SFT or shared_state RLCD")
    return validate_schedule(config)


def run(config, resume=None, initialize=None):
    config = normalize_config(config)
    if resume and initialize:
        raise ValueError("Use either resume or initialize")
    if config["method"] == "rlcd" and not (resume or initialize):
        raise ValueError("RLCD requires --initialize from an SFT checkpoint or --resume")
    if initialize and Path(config["output"]).resolve() == Path(initialize).resolve().parent:
        raise ValueError("Initialization requires a new output directory")
    context = initialize_distributed(config.get("device", "cuda"))
    config = dict(config, device=context.device)
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + context.rank)
    data_path = Path(config["data"])
    backend = get_backend(config["architecture"])
    records = read_jsonl(data_path, candidate_fn=backend.candidates)
    config = dict(config, data_sha256=file_sha256(data_path))
    fingerprints = context.gather({k: v for k, v in config.items() if k != "device"})
    if any(value != fingerprints[0] for value in fingerprints):
        raise ValueError("All ranks must use identical config and dataset contents")
    model = build_model(config)
    if context.primary:
        print("Loaded Model!", flush=True)
    groups = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", 0.01))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    progress = {"epoch": 0, "cursor": 0, "step": 0, "order": None}
    payload = None
    if initialize:
        payload = load_checkpoint(initialize, model, strict=False)
        previous = payload["config"]
        previous = normalize_model_config(previous)
        backend.validate_initialization(previous, config)
    if resume:
        payload = load_checkpoint(resume, model, optimizer, rng, rank=context.rank, world_size=context.world_size)
        allowed_changes = {"output", "epochs", "max_steps", "device", "save_every",
                           "microbatch_size", "microbatch_max_tokens", "data_prefetch"}
        if config["architecture"] == "qwen":
            allowed_changes.update({"attn_implementation", "gradient_checkpointing"})
        previous = normalize_config(payload["config"])
        if {k: v for k, v in previous.items() if k not in allowed_changes} != {k: v for k, v in config.items() if k not in allowed_changes}:
            raise ValueError("Resume config/data changed; use --initialize for a new stage")
        progress = payload["progress"]
    print(f"[rank {context.rank}] broadcast begin", flush=True)
    broadcast_trainable(model, context)
    print(f"[rank {context.rank}] broadcast done", flush=True)
    continual = None
    if config.get("finetuning_type") == "cl_moe":
        from .cl_moe import ContinualMoE
        continual = ContinualMoE(model, config, payload, resuming=bool(resume))
    if config["method"] == "rlcd":
        if not resume:
            torch.manual_seed(seed + context.rank)
        objective = RLCDObjective(model, config["rlcd"],
                                  payload.get("training_state") if resume else None, resuming=bool(resume),
                                  cache_frozen_features=config.get("cache_frozen_features", False),
                                  batch_config=config)
    else:
        objective = SFTObjective(config)
    progress.setdefault("optimizer_steps", progress["step"])
    schedule = LearningRateSchedule(config, optimizer, len(records), context.world_size)
    compiler = build_compiler(config, data_path.parent)
    report = {"config": config, "world_size": context.world_size, "gradient_sync": "sum_after_local_accumulation",
              "initialization_checkpoint": str(initialize) if initialize else None,
              "initialization_sha256": file_sha256(Path(initialize) / "checkpoint.pt") if initialize else None,
              "tokens_per_step_scope": "per_rank", "adaptation": backend.adaptation(model, config),
              "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
              "optimizer_groups": [{"name": g["name"], "lr": g["lr"], "parameters": sum(p.numel() for p in g["params"])} for g in groups],
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "av")}}
    report["learning_rate_schedule"] = schedule.metadata()
    if context.primary:
        if resume:
            previous_manifest = Path(resume).parent / "run_manifest.json"
            if previous_manifest.exists():
                origin = json.loads(previous_manifest.read_text(encoding="utf-8"))
                for key in ("initialization_checkpoint", "initialization_sha256"):
                    report[key] = origin.get(key)
            report["resume_checkpoint"] = str(resume)
            report["resume_step"] = progress["step"]
            report["resume_runtime_changes"] = {
                key: {"previous": previous.get(key), "current": config.get(key)}
                for key in ("attn_implementation", "gradient_checkpointing", "microbatch_size", "microbatch_max_tokens", "data_prefetch")
                if previous.get(key) != config.get(key)}
        (output / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"world_size": context.world_size, "optimizer_groups": report["optimizer_groups"]}), flush=True)
    max_steps = config.get("max_steps", 2**63-1)
    token_budget = config.get("tokens_per_step", 16384)
    if token_budget <= 0:
        raise ValueError("tokens_per_step must be positive")
    def save():
        rank_states = context.gather(capture_rank_state(progress, rng))
        if context.primary:
            training_state = objective.state_dict()
            if continual is not None:
                training_state = {"cl_moe": continual.state_dict()}
            save_checkpoint(output / "latest", model, optimizer, config, progress, rng,
                            rank_states=rank_states, training_state=training_state)
        context.barrier()

    model.train()
    if config["method"] == "rlcd":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.eval()
    started = time.monotonic()
    prior_elapsed = 0.0
    if resume and (output / "metrics.jsonl").exists():
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["step"] == progress["step"]:
                prior_elapsed = row.get("elapsed_seconds", 0.0)
    session_states = session_questions = 0
    metric_file = (output / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8") if context.primary else nullcontext(None)
    media_name = "media.jsonl" if context.world_size == 1 else f"media.rank{context.rank}.jsonl"
    # 每个 rank 都执行这段训练流程，分别持有自己的模型、优化器和数据分片。
    # metrics 只由 rank 0 写；media 日志每个 rank 单独写；退出 with 时关闭文件和预取器。
    if context.primary:
        print("========== Training ==========", flush=True)
        print("Timing (seconds, last completed step): D=data wait, F=forward+loss, "
              "B=backward, S=gradient sync, O=optimizer, X=other stages, T=step total; "
              "u=training units. CUDA phase timing synchronizes the device. "
              "T excludes reporting and checkpoint saving.", flush=True)
    read_bars = []
    read_bar_epoch = None
    trace_steps, trace_timeout = trace_options()
    with metric_file as log, (output / media_name).open("a" if resume else "w", encoding="utf-8") as media_log, make_prefetcher(records, compiler, backend, config, context.world_size) as pack_source, ExitStack() as read_bar_stack:
        # epoch 表示已完成的数据遍历轮数；step 表示已处理的全局 pack 批次数。
        # 每次 while 只取一批数据，不是执行一个完整 epoch；任一上限到达即结束。
        while progress["epoch"] < config.get("epochs", 1) and progress["step"] < max_steps:
            if progress["order"] is None:
                # 新 epoch 才生成顺序；resume 时保留 checkpoint 中的 order 和 cursor。
                if context.world_size == 1:
                    progress["order"] = list(range(len(records)))
                    rng.shuffle(progress["order"])
                else:
                    # 所有 rank 使用相同的全局打乱顺序，再按 rank 切片，各取不同样本。
                    progress["order"] = epoch_shard(len(records), seed, progress["epoch"], context.rank, context.world_size)
            # 所有 rank 提交读取位置，仅 rank 0 绘制多行进度，避免跨进程刷新终端。
            # 在取本批前显示恢复位置；取完后再次更新，100% 仅表示读取完毕。
            if read_bar_epoch != progress["epoch"]:
                read_positions = context.gather({"rank": context.rank, "cursor": progress["cursor"],
                                                 "total": len(progress["order"])})
                if context.primary:
                    read_bar_stack.close()
                    read_bars = [read_bar_stack.enter_context(tqdm(
                        total=row["total"], initial=row["cursor"],
                        desc=f"Epoch {progress['epoch']} | rank {row['rank']} read",
                        position=row["rank"], unit="record", dynamic_ncols=True,
                    )) for row in read_positions]
                read_bar_epoch = progress["epoch"]
            timer = StepTimer(context.device, PhaseTrace(context.rank, progress["step"] + 1,
                enabled=progress["step"] < trace_steps, timeout=trace_timeout))
            training_units = 0
            prepare_started = time.monotonic()
            # 从本 rank 的 cursor 开始取一个累积批次：按 token 预算或固定 batch 组包。
            # 开启预取时可接收后台已准备好的结果；达到最后一步时不再预取下一批。
            with timer.measure("data"):
                prepared = pack_source.take(progress["order"], progress["cursor"], rng.getstate(),
                                            prefetch_next=progress["step"] + 1 < max_steps)
            # 此处计量主线程等待取批的时间，可能小于后台实际 CPU 准备时间。
            prepare_wait_seconds = time.monotonic() - prepare_started
            # pack 是本轮本卡的有效 state 集合；一个 state 可以包含多个 question。
            # tokens 是这些 state 的计算预算计数；cursor 是原始分片中已扫描的位置。
            pack, tokens = prepared.states, prepared.tokens
            if timer.trace.enabled:
                timer.trace.emit("pack", "ready", states=len(pack), compute_tokens=tokens,
                    sequence_lengths=[s.inputs["input_ids"].shape[-1] for s in pack
                                      if getattr(s, "inputs", None) is not None])
            # 更新当前的cursor
            progress["cursor"] = prepared.cursor
            # 仅接收本次取出的 pack 对应的随机数状态，保持恢复训练的数据处理可复现。
            rng.setstate(prepared.rng_state)
            for event in prepared.media:
                # 记录图片/视频等媒体处理事件；此时 step 尚未递增，因此标记为 step + 1。
                media_log.write(json.dumps({"epoch": progress["epoch"], "step": progress["step"] + 1,
                                            "rank": context.rank, **event}, default=str) + "\n")
            # 四项本地统计：state 数、question 数、token 数、是否读完本 rank 当前 epoch。
            # 完成标志为 0/1，求和后就是已经读到分片末尾的 rank 数量。
            local_counts = [len(pack), sum(len(s.questions) for s in pack), tokens,
                            int(progress["cursor"] == len(progress["order"]))]
            # 所有 rank 必须参与这次通信，并且都会得到相同的全局计数。
            with timer.measure("counts"):
                counts = context.sum(local_counts)
                global_states, global_questions, global_tokens, finished_ranks = map(int, counts)
                read_positions = context.gather(progress["cursor"])
            if context.primary:
                for bar, cursor in zip(read_bars, read_positions):
                    # 读取新 pack 时保留上一个已完成 step 的耗时，直到本轮结束。
                    bar.update(cursor - bar.n)
                    bar.refresh()
            # 只要全局还有有效数据，所有 rank 都进入更新流程，即使本卡 pack 为空。
            # 空闲 rank 可以没有本地 backward，但仍必须参与后面的梯度同步和参数更新。
            if global_states:
                # question_mean：每题等权；state_mean：每个 state 等权，内部题目均分权重。
                # 分母使用全局数量，保证各卡数据量不同时仍得到正确的全局平均梯度。
                normalization = global_questions if config["loss_reduction"] == "question_mean" else global_states
                # SFT 返回占位数据；RLCD 在这里采样并固定 rollout，供本批多次优化复用。
                with timer.measure("prepare"):
                    rollouts = objective.prepare(model, pack)
                # accumulated 用于汇总 num_iterations 次更新的平均日志指标，不保存梯度。
                accumulated = {name: 0.0 for name in ("loss", *objective.metric_names)}
                # 同一个 pack 可以连续优化多次；每次都重新前向/反向并 optimizer.step。
                # 当前固定 global_batch_size 的 SFT 模式要求 num_iterations 为 1。
                for _ in range(objective.num_iterations):
                    # 每次优化开始时清空梯度；下面各 unit 的 backward 会累积到同一份 .grad。
                    with timer.measure("zero_grad"):
                        optimizer.zero_grad(set_to_none=True)
                    stats = dict.fromkeys(accumulated, 0.0)
                    # training_batches 按 microbatch/后端组织前向，yield 的 unit 含决策 logits。
                    # 各 rank 的 unit 数量可以不同；next(units) 本身执行前向，必须计时。
                    units = iter(training_batches(model, backend, pack, rollouts, config))
                    while True:
                        with timer.measure("forward"):
                            try:
                                unit = next(units)
                            except StopIteration:
                                break
                            training_units += 1
                            # weight 由组批器给出：题目平均时为 1，state 平均时为 1/该 state 题数。
                            # 再除以全局 normalization，得到每道题对全局目标的贡献。
                            losses = [(objective.loss_from_logits(d.logits, d.question, d.rollout), weight / normalization)
                                      for d, weight in unit]
                            # 同一 unit 内的题目可能共享计算图，合并损失后统一反向一次。
                            combined = torch.stack([loss * scale for (loss, _), scale in losses]).sum()
                            if not torch.isfinite(combined):
                                raise FloatingPointError("Non-finite training unit loss")
                        with timer.measure("backward"):
                            combined.backward()
                        # 提取普通数值累计日志，不把计算图保留在 stats 中。
                        with timer.measure("metrics"):
                            for (loss, terms), scale in losses:
                                stats["loss"] += loss.item() * scale
                                for name, value in terms.items():
                                    stats[name] += value.item() * scale
                    # 本卡全部 unit 反向完成后，再对各卡梯度求和。损失已经按全局数量
                    # 归一化，所以这里不再除以 world_size；本卡未用到的参数也由同步逻辑处理。
                    with timer.measure("sync"):
                        synchronize_gradients(model, context)
                    # 日志指标采用相同的全局求和，得到本次优化的全局 loss 及各损失项。
                    with timer.measure("metrics"):
                        stats = dict(zip(stats, context.sum(list(stats.values()))))
                    # 记录同步后、裁剪前的各参数组梯度范数，便于检查训练是否异常。
                    with timer.measure("clip"):
                        group_norms = {g["name"]: float(torch.stack([p.grad.detach().float().square().sum() for p in g["params"] if p.grad is not None]).sum().sqrt())
                                       for g in groups if any(p.grad is not None for p in g["params"])}
                        # 对所有可训练参数统一裁剪；返回的 norm 是裁剪前的总范数。
                        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.get("max_grad_norm", 1.0), error_if_nonfinite=True)
                    # 先按实际优化次数设置本次学习率，再更新每个 rank 自己的模型副本。
                    with timer.measure("optimizer"):
                        schedule.apply(progress["optimizer_steps"])
                        optimizer.step()
                        progress["optimizer_steps"] += 1
                    # 日志对同一 pack 的多次优化取平均；不是对更新后的模型参数取平均。
                    for name, value in stats.items():
                        accumulated[name] += value / objective.num_iterations
                stats = accumulated
                # 一个 pack 只增加一个 step；optimizer_steps 则增加 num_iterations 次。
                progress["step"] += 1
                # total 截止到更新完成，不含下方报告 gather、日志写盘和 checkpoint 保存。
                timing_seconds = timer.snapshot()
                # 所有 rank 都参加 gather，收集各卡的数据进度、准备耗时和显存统计。
                # peak_cuda_bytes 是 CUDA 分配器自上次重置后的峰值，这里没有逐 step 重置。
                rank_counts = context.gather({"rank": context.rank, "states": len(pack), "compute_tokens": tokens,
                                              "cursor": progress["cursor"],
                                              "prepare_wait_seconds": prepare_wait_seconds,
                                              "cpu_prepare_seconds": prepared.prepare_seconds,
                                              "timing_seconds": timing_seconds, "training_units": training_units,
                                              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
                if context.primary:
                    for bar, row in zip(read_bars, rank_counts):
                        bar.set_postfix_str(timing_postfix(progress["step"], row["timing_seconds"],
                                                         row["training_units"]))
                # 本次进程启动以来处理的有效样本/问题数，用于计算平均吞吐量。
                session_states += global_states
                session_questions += global_questions
                elapsed = time.monotonic() - started
                # cursor 统计已扫描原始记录，可能包含被过滤、未成为有效 state 的记录。
                # 跨 rank 求和得到本 epoch 进度，加上已完成 epoch 得到总进度。
                epoch_records = sum(r["cursor"] for r in rank_counts)
                completed_records = progress["epoch"] * len(records) + epoch_records
                total_records = config.get("epochs", 1) * len(records)
                # 总耗时含 resume 前的已记录时间；吞吐量使用本次启动的计数和耗时。
                # ETA 是按当前平均吞吐估计的本 epoch 剩余时间；total_records 按 epochs
                # 计算，未按 max_steps 提前停止截断。多次优化时梯度范数来自最后一次更新。
                stats.update(learning_rates={g["name"]: g["lr"] for g in optimizer.param_groups},
                             step=progress["step"], optimizer_steps=progress["optimizer_steps"],
                             method=config["method"], epoch=progress["epoch"], states=global_states,
                             questions=global_questions, compute_tokens=global_tokens, ranks=rank_counts,
                             grad_norm=float(norm), group_grad_norms=group_norms,
                             peak_cuda_bytes=max(r["peak_cuda_bytes"] for r in rank_counts),
                             completed_records=completed_records, total_records=total_records,
                             epoch_fraction=epoch_records / len(records), elapsed_seconds=prior_elapsed + elapsed,
                             states_per_second=session_states / elapsed,
                             questions_per_second=session_questions / elapsed,
                             eta_epoch_seconds=(len(records) - epoch_records) * elapsed / session_states)
                if context.primary:
                    # 每个全局 step 由 rank 0 写一行 JSON 并刷新。当前终端 JSON 打印已注释。
                    log.write(json.dumps(stats) + "\n")
                    log.flush()
                    # print(json.dumps(stats), flush=True)
            # 各卡都读完本 epoch 后统一推进；即使本轮全局没有有效 state，也检查此条件。
            # 重置 cursor 和 order，使下一轮重新打乱/分片。先读完的 rank 不能独自切换 epoch。
            if finished_ranks == context.world_size:
                progress.update(epoch=progress["epoch"] + 1, cursor=0, order=None)
            if global_states and progress["step"] % config.get("save_every", 100) == 0:
                media_log.flush()
                # save 内含 gather/barrier，必须所有 rank 调用；只有 rank 0 实际写 latest。
                save()
        if progress["step"] == 0:
            raise ValueError("No optimizer step: dataset has no labels or epochs/max_steps is zero")
        # 循环正常结束时额外保存一次，确保最后不足 save_every 间隔的进度也被保存。
        save()
    if continual is not None:
        # CL-MoE 的阶段收尾：用配置指定的训练样本统计路由/任务原型，再执行专家动量融合。
        # 这不是上面的梯度优化循环；所有 rank 仍需参与其中的统计通信。
        if context.primary:
            print("========== Train End ==========", flush=True)
        report = continual.consolidate(model, compiler, backend, records, context)
        rank_states = context.gather(capture_rank_state(progress, rng))
        if context.primary:
            # consolidated 用于下一任务的 initialize；当前任务断点 resume 使用融合前 latest。
            save_checkpoint(output / "consolidated", model, optimizer, config, progress, rng,
                            rank_states=rank_states, training_state={"cl_moe_consolidated": True})
            (output / "consolidation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # 等待主进程完成保存，所有 rank 再结束当前阶段。
        context.barrier()
    return model, compiler, progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--attn-implementation", choices=ATTENTION_IMPLEMENTATIONS,
                        help="Override the Qwen attention backend in the training config")
    parser.add_argument("--method", choices=("sft", "rlcd"), help="Override the training objective (default: SFT)")
    parser.add_argument("--output", help="Override output directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--initialize")
    args = parser.parse_args()
    try:
        config = normalize_model_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
        config = with_attention_implementation(config, args.attn_implementation)
        if args.method:
            config["method"] = args.method
        if args.output:
            config["output"] = args.output
        run(config, args.resume, args.initialize)
    finally:
        close()


if __name__ == "__main__":
    main()
