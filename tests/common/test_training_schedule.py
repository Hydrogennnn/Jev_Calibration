from copy import deepcopy
import random
from types import SimpleNamespace

import pytest
import torch

from valen.training.distributed import epoch_shard
from valen.training.schedule import LearningRateSchedule, make_prefetcher, validate_schedule


def test_cosine_warmup_and_optimizer_reload_keep_identical_updates():
    config = {"global_batch_size": 4, "epochs": 1, "lr_scheduler": "cosine", "warmup_ratio": .2}
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    schedule = LearningRateSchedule(config, optimizer, 40, 2)
    assert schedule.steps == 10 and schedule.warmup == 2
    assert [schedule.factor(i) for i in (0, 1, 2, 6, 10)] == pytest.approx([0, .5, 1, .5, 0])
    for step in range(4):
        parameter.grad = torch.ones_like(parameter)
        schedule.apply(step)
        optimizer.step()
    restored = torch.nn.Parameter(parameter.detach().clone())
    other_optimizer = torch.optim.AdamW([restored], lr=.1)
    other_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    other_schedule = LearningRateSchedule(config, other_optimizer, 40, 2)
    for step in range(4, 10):
        for p, opt, sched in ((parameter, optimizer, schedule), (restored, other_optimizer, other_schedule)):
            p.grad = torch.ones_like(p)
            sched.apply(step)
            opt.step()
        torch.testing.assert_close(parameter, restored, rtol=0, atol=0)


@pytest.mark.parametrize("prefetch", [False, True])
def test_fixed_batches_cover_uneven_shards_without_duplicates(prefetch):
    records = [{"group_id": str(i)} for i in range(17)]
    compiler = SimpleNamespace(compile=lambda record, rng, **kw: SimpleNamespace(
        compute_tokens=2, questions=[record["group_id"]], media=[]))
    backend = SimpleNamespace(validate_state=lambda *args: None)
    config = {"global_batch_size": 8, "tokens_per_step": 10, "data_prefetch": prefetch}
    sizes, seen = [], []
    for rank in range(4):
        order = epoch_shard(len(records), 42, 0, rank, 4)
        cursor, rng, counts = 0, random.Random(42), []
        with make_prefetcher(records, compiler, backend, config, 4) as source:
            while cursor < len(order):
                pack = source.take(order, cursor, rng.getstate())
                cursor = pack.cursor
                rng.setstate(pack.rng_state)
                counts.append(len(pack.states))
                seen.extend(state.questions[0] for state in pack.states)
            empty = source.take(order, cursor, rng.getstate())
            assert not empty.states
        sizes.append(counts)
    assert sizes == [[2, 2, 1], [2, 2], [2, 2], [2, 2]]
    assert sorted(map(int, seen)) == list(range(17))
    with pytest.raises(ValueError, match="truncated"):
        with make_prefetcher(records, compiler, backend, dict(config, tokens_per_step=2), 4) as source:
            source.take(list(range(17)), 0, random.Random(0).getstate())


def test_legacy_constant_schedule_and_invalid_configuration():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.01)
    before = deepcopy(optimizer.state_dict())
    LearningRateSchedule({}, optimizer, 3, 1).apply(100)
    assert optimizer.state_dict() == before
    assert validate_schedule({}) == {}
    for config in ({"lr_scheduler": "cosine"}, {"global_batch_size": 0},
                   {"warmup_ratio": .1}, {"global_batch_size": 4, "method": "rlcd"},
                   {"global_batch_size": 4, "sft_iterations": 2}):
        with pytest.raises(ValueError):
            validate_schedule(config)
    with pytest.raises(ValueError, match="divisible"):
        LearningRateSchedule({"global_batch_size": 4}, optimizer, 3, 8)
