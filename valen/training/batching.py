"""Batch independent QAs or shared states. / 按配置保留 state 或 QA 的损失权重。"""
from valen.modeling.interfaces import Decision


def question_weight(state, config):
    return 1.0 if config.get("loss_reduction", "state_mean") == "question_mean" else 1 / len(state.questions)


"""
class PreparedPack:
    states: list
    tokens: int
    cursor: int
    rng_state: tuple
    media: list
    prepare_seconds: float

"""

"""
@dataclass
class CompiledState:
    questions: list
    logical_tokens: int
    compute_tokens: int
    media: list
    inputs: dict | None = None
"""




def training_batches(model, backend, pack, rollouts, config):
    """将一个梯度累积 pack 转为训练批次，逐批产出 (Decision, 损失权重) 列表。

    rollouts 与 pack 中的 state 一一对应；组批时保留各题目所属 state 的权重
    这里的pack对应这里的 PreparedPack.states, list of CompiledState
    """
    size = config.get("microbatch_size", 1)
    if len(pack) != len(rollouts):
        raise ValueError("Rollout/state count mismatch")
    # 不合并微批次时，由后端决定训练单元边界，保留其计算图与反向传播语义。
    if size == 1:
        for state, state_rollouts in zip(pack, rollouts):
            weight = question_weight(state, config)
            for unit in backend.training_units(model, state, state_rollouts):
                yield [(decision, weight) for decision in unit]
        return
    # 并行微批次仅支持 Qwen；RLCD 还要求使用下方的共享 state 输入路径。
    method = config.get("method", "sft")
    if backend.name != "qwen" or method not in {"sft", "rlcd"}:
        raise ValueError("Parallel microbatches require Qwen SFT or shared-state RLCD")
    # state.inputs 表示整组题目共用输入；这类输入按 state 组批，不能混入独立题目输入。
    shared = [getattr(state, "inputs", None) is not None for state in pack]
    if any(shared):
        if not all(shared):
            raise ValueError("Cannot mix question and shared_state inputs in one microbatch")
        yield from shared_state_batches(model, pack, rollouts, config)
        return
    if method != "sft":
        raise ValueError("RLCD microbatches require shared_state inputs")
    max_tokens = config.get("microbatch_max_tokens", 32768)
    # 独立题目的 SFT 路径：校验 rollout 均为 None，再展开为带原始损失权重的题目。
    # question_mean 下每题权重为 1；state_mean 下每题权重为所属 state 题目数的倒数。
    entries = []
    for state, state_rollouts in zip(pack, rollouts):
        if len(state.questions) != len(state_rollouts):
            raise ValueError("Rollout/question count mismatch")
        for question, rollout in zip(state.questions, state_rollouts):
            if rollout is not None:
                raise ValueError("Qwen SFT microbatches require empty rollouts")
            entries.append((question, question_weight(state, config)))
    # Sort only within the existing accumulation pack; data sharding stays fixed.
    # 仅在本步内按长度分组，减少 padding，不改变数据分片和每步 QA 数量。
    entries.sort(key=lambda entry: max(b.inputs["input_ids"].shape[1] for b in entry[0].branches))
    # batch 记录题目及权重；branches 是分支总数，longest 是最长分支的 token 数。
    batch, branches, longest = [], 0, 0

    def forward(items):
        # 批量前向后，按题目顺序封装 logits，并附回组批前的损失权重。
        questions = [question for question, _ in items]
        logits = model.forward_batch(questions, max_tokens=max_tokens)
        return [(Decision(question, output), weight)
                for (question, weight), output in zip(items, logits)]

    for question, weight in entries:
        lengths = [b.inputs["input_ids"].shape[1] for b in question.branches]
        # 估算加入当前题目后的 padding 开销：最长分支长度 × 全部题目的分支总数。
        padded_tokens = max(longest, max(lengths)) * (branches + len(lengths))
        # 题目数已达上限，或加入后超出 token 预算时，先提交已有批次。
        # 单个题目不在此处拆分；空批次仍接收它，分支的实际组批交由模型处理。
        if batch and (len(batch) >= size or padded_tokens > max_tokens):
            yield forward(batch)
            batch, branches, longest = [], 0, 0
        batch.append((question, weight))
        branches += len(lengths)
        longest = max(longest, max(lengths))
    # 提交末尾尚未产出的批次。
    if batch:
        yield forward(batch)


def shared_state_indices(pack, config):
    """Group state indices without changing identities. / 按长度组批，保留 state 身份。"""
    size, max_tokens = config["microbatch_size"], config.get("microbatch_max_tokens", 32768)
    indices = sorted(range(len(pack)), key=lambda i: pack[i].inputs["input_ids"].shape[1])
    batch, longest = [], 0
    for index in indices:
        length = pack[index].inputs["input_ids"].shape[1]
        if batch and (len(batch) >= size or max(longest, length) * (len(batch) + 1) > max_tokens):
            yield batch
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        yield batch


def shared_state_batches(model, pack, rollouts, config):
    """每个 state 只占一行；所有题目共用一次反向。 / Never duplicate shared graphs per QA."""
    max_tokens = config.get("microbatch_max_tokens", 32768)
    for state, group in zip(pack, rollouts):
        if len(state.questions) != len(group):
            raise ValueError("Rollout/question count mismatch")
        if config.get("method", "sft") == "sft" and any(rollout is not None for rollout in group):
            raise ValueError("Qwen SFT microbatches require empty rollouts")
        if any(getattr(rollout, "features", None) is not None for rollout in group):
            raise ValueError("Batched shared-state updates do not support frozen feature caches")
    for indices in shared_state_indices(pack, config):
        items = [pack[index] for index in indices]
        logits = model.forward_state_batch(items, max_tokens=max_tokens)
        yield [(Decision(question, output, rollout), question_weight(state, config))
               for index, state, outputs in zip(indices, items, logits)
               for question, output, rollout in zip(state.questions, outputs, rollouts[index])]
