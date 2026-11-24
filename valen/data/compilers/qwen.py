"""Compile Qwen question branches or a shared sequence with per-question readouts."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import torch
from valen.data.schema import candidates, target_distribution
from valen.data.types import QuestionSpec


@dataclass
class Branch:
    inputs: dict
    candidate_positions: list
    decision_position: int
    context_span: tuple | None = None
    instruction_span: tuple | None = None
    candidate_spans: list | None = None


@dataclass(init=False)
class Question(QuestionSpec):
    branches: list

    def __init__(self, qid, kind, keys, descriptions, branches, target, is_text):
        # Keep the positional constructor used by existing compiler clients.
        super().__init__(qid, kind, keys, descriptions, target, is_text)
        self.branches = branches


@dataclass
class CompiledState:
    questions: list
    logical_tokens: int
    compute_tokens: int
    media: list
    inputs: dict | None = None


class Compiler:
    def __init__(self, processor, media_root=".", max_length=8192, media_kwargs=None, verified_media_hashes=None,
                 execution="question"):
        if execution not in {"question", "shared_state"}:
            raise ValueError(f"Unknown qwen_execution: {execution}")
        self.execution = execution
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.media_root = Path(media_root)
        self.max_length = max_length
        self.media_kwargs = media_kwargs or {}
        # Optional, pre-audited immutable assets for latency measurements.
        # Normal training/inference continues to hash each input at use time.
        self.verified_media_hashes = verified_media_hashes or {}

    def _messages(self, state):
        messages = [{"role": "user", "content": state}] if isinstance(state, str) else deepcopy(state["messages"])
        media = []
        for message in messages:
            if message["role"] not in {"user", "assistant", "system"}:
                raise ValueError("Unsupported state role")
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            for item in message["content"]:
                kind = item["type"]
                if kind == "text":
                    # Prevent user text from injecting Qwen control tokens.
                    if any(t in item["text"] for t in self.tokenizer.all_special_tokens):
                        raise ValueError("Reserved tokenizer control token in state")
                    continue
                if kind not in {"image_url", "video_url"}:
                    raise ValueError(f"Unsupported content type: {kind}")
                url = item[kind]["url"]
                if "://" in url:
                    raise ValueError("Materialize remote media locally with a hash before training")
                path = Path(url)
                path = path if path.is_absolute() else self.media_root / path
                path = path.resolve(strict=True)
                digest = self.verified_media_hashes.get(str(path))
                if digest is None:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                media.append({"path": str(path), "sha256": digest, "type": kind})
                item.clear()
                item.update(type=kind.removesuffix("_url"), path=str(path))
        return messages, media

    def compile(self, record, rng=None, labeled_only=False):
        # 将单条 record 编译为 CompiledState：按配置构建独立题目分支或共享输入。
        # 此处只准备模型输入、标签和位置索引，不执行模型前向计算。
        selected = []
        for qid, question in record["request"]["questions"].items():
            pairs = candidates(question)  # 候选项列表，每项为 (key, 描述)。
            target = record.get("targets", {}).get(qid)
            # 独立题目模式可直接跳过无标签题；共享模式仍需保留其提示文本。
            if labeled_only and self.execution == "question" and target_distribution(target, [k for k, _ in pairs]) is None:
                continue
            # 随机化 choice 的展示顺序；key 与描述一起移动，标签稍后按新 keys 顺序对齐。
            if rng is not None and question["type"] == "choice":
                rng.shuffle(pairs)
            selected.append((qid, question, pairs, target))
        # 标签只决定监督范围，共享序列仍保留所有题。 / Labels select losses, never prompt content.
        # 没有题目，或仅监督模式下没有任何有效标签时，无需继续处理媒体。
        if not selected or (labeled_only and not any(
                target_distribution(target, [k for k, _ in pairs]) is not None
                for _, _, pairs, target in selected)):
            return CompiledState([], 0, 0, [])
        # 将 state 转成消息及本地媒体路径，并核对 assets 中声明的媒体哈希。
        messages, media = self._messages(record["request"]["state"])
        expected_assets = {str((self.media_root / a["path"]).resolve()): a["sha256"] for a in record.get("assets", [])}
        for entry in media:
            if entry["path"] in expected_assets and expected_assets[entry["path"]] != entry["sha256"]:
                raise ValueError(f"Media hash mismatch: {entry['path']}")
        # 套用对话模板、分词并预处理媒体，得到公共上下文张量；尚未追加题目内容。
        base = dict(self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors="pt", processor_kwargs=dict(self.media_kwargs, return_metadata=True)))
        # Metadata is audit information, never a backbone keyword argument.
        # 视频采样元数据移入 media 留作审计，避免作为模型参数传入 backbone。
        metadata = base.pop("video_metadata", None)
        if metadata is not None:
            for entry, meta in zip([m for m in media if m["type"] == "video_url"], metadata):
                entry["sampling"] = dict(meta) if isinstance(meta, dict) else vars(meta)
                sampling = entry["sampling"]
                indices = sampling.get("frames_indices")
                if indices is not None and sampling.get("fps"):
                    # 将采样帧索引换算成秒，并记录最早与最晚采样帧的时间。
                    sampling["frames_indices"] = [int(i) for i in indices]
                    sampling["timestamps_seconds"] = [int(i) / sampling["fps"] for i in indices]
                    sampling["visible_range_seconds"] = [min(sampling["timestamps_seconds"]), max(sampling["timestamps_seconds"])]
        # 记录视觉网格及 token 数；此处按每 4 个网格单元合并为 1 个视觉 token 计数。
        for name in ("image_grid_thw", "video_grid_thw"):
            if name in base:
                grid = base[name]
                media.append({name: grid.tolist(), "visual_tokens": int(grid.prod(-1).sum()) // 4})
        base_length = base["input_ids"].shape[1]
        # 共享模式将所有题目拼入同一序列；下面的逻辑只处理独立题目模式。
        if self.execution == "shared_state":
            return self._compile_shared(selected, base, media, labeled_only)
        # 逻辑 token 数仅计一次 base；计算 token 数则对每个分支重复计入 base。
        logical_tokens, compute_tokens = base_length, 0
        compiled = []
        for qid, question, pairs, target in selected:
            keys = [k for k, _ in pairs]
            branches = []
            # score 每个等级单独一个分支；choice/noul 在同一分支中放入全部候选项。
            groups = [[pair] for pair in pairs] if question["type"] == "score" else [pairs]
            for group in groups:
                suffix = []
                def append(text, char_span=None):
                    # 追加文本 token；若提供字符区间，返回其在 base + suffix 中的 token 区间。
                    start = base_length + len(suffix)
                    if char_span is None:
                        suffix.extend(self.tokenizer.encode(text, add_special_tokens=False))
                        return None
                    # 保持原有分词，只额外记录区间。 / Preserve token IDs; record role spans.
                    encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                    suffix.extend(encoded["input_ids"])
                    # 选取与字符区间有交集的 token，最终返回左闭右开的绝对位置区间。
                    indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                               if a < char_span[1] and b > char_span[0]]
                    if not indices:
                        raise ValueError("Empty token span for decision role")
                    return (start + indices[0], start + indices[-1] + 1)
                # Canonical segment tokenization makes endpoints exact; no string searching.
                # 当前分支的题目作为新的 user 消息追加，按片段分词以直接记录边界。
                suffix.extend(self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
                # 拒绝题干、候选 key 或描述中的保留控制 token，防止破坏消息结构。
                for value in [question["instructions"]] + [v for pair in group for v in pair]:
                    if any(t in value for t in self.tokenizer.all_special_tokens):
                        raise ValueError("Reserved tokenizer control token in question/criteria")
                prefix = "Task: " + question["type"] + "\nQuestion: "
                # 写入任务类型和题干，但 instruction_span 只覆盖题干正文。
                instruction_span = append(prefix + question["instructions"] + "\nCandidates:\n",
                                          (len(prefix), len(prefix) + len(question["instructions"])))
                positions, candidate_spans = [], []
                for key, description in group:
                    # Score never sees its level ID, nor any other level description.
                    # score 只写等级描述；其他题型写入 key 和描述，span 仅覆盖描述正文。
                    prefix = key + ": " if question["type"] != "score" else ""
                    candidate_spans.append(append(prefix + description, (len(prefix), len(prefix) + len(description))))
                    # 在追加换行之前，记录该候选描述最后一个 token 的位置。
                    positions.append(base_length + len(suffix) - 1)
                    append("\n")
                # 固定决策标记的最后一个 token 用作读出位置；输入中不追加真实答案。
                append("Decision:")
                length = base_length + len(suffix)
                # 超长直接报错，避免截断导致候选内容或已记录的位置失效。
                if length > self.max_length:
                    raise ValueError(f"{qid}: {length} tokens exceed max_length={self.max_length}; no truncation")
                # 浅复制 base，复用媒体张量等字段，再生成当前分支独立的 token 序列。
                inputs = dict(base)
                inputs["input_ids"] = torch.cat([base["input_ids"], torch.tensor([suffix], dtype=torch.long)], dim=1)
                # 当前分支按完整有效序列构建掩码；新增题目文本的多模态类型标记为 0。
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                if "mm_token_type_ids" in base:
                    inputs["mm_token_type_ids"] = torch.cat([base["mm_token_type_ids"], torch.zeros((1, len(suffix)), dtype=torch.long)], dim=1)
                # 保存输入、候选读出位置、决策位置，以及上下文/题干/候选描述的区间。
                branches.append(Branch(inputs, positions, length - 1, (0, base_length),
                                       instruction_span, candidate_spans))
                compute_tokens += length
                logical_tokens += len(suffix)
            # 按当前 keys 顺序生成标签概率向量，确保 shuffle 后候选项与监督仍然对应。
            compiled.append(Question(qid, question["type"], keys, [v for _, v in pairs], branches,
                                     target_distribution(target, keys), not media))
        return CompiledState(compiled, logical_tokens, compute_tokens, media)

    def _compile_shared(self, selected, base, media, labeled_only=False):
        """All schemas precede all readouts; labels and external IDs never enter inputs.

        This is ordinary causal attention, not isolated question/Score branches.
        No answer tokens or LM output layer are required by the decision heads.
        """
        base_length = base["input_ids"].shape[1]
        suffix, compiled = [], []

        def append(text, span=None):
            start = base_length + len(suffix)
            if span is None:
                suffix.extend(self.tokenizer.encode(text, add_special_tokens=False))
                return None
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            suffix.extend(encoded["input_ids"])
            indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < span[1] and b > span[0]]
            if not indices:
                raise ValueError("Empty token span for decision role")
            return start + indices[0], start + indices[-1] + 1

        append("<|im_start|>user\n")
        for index, (qid, question, pairs, target) in enumerate(selected):
            for value in [question["instructions"]] + [v for pair in pairs for v in pair]:
                if any(t in value for t in self.tokenizer.all_special_tokens):
                    raise ValueError("Reserved tokenizer control token in question/criteria")
            prefix = f"Question {index + 1}\nTask: {question['type']}\nQuestion: "
            instruction = append(prefix + question["instructions"] + "\nCandidates:\n",
                                 (len(prefix), len(prefix) + len(question["instructions"])))
            positions, spans = [], []
            for key, description in pairs:
                prefix = key + ": " if question["type"] != "score" else ""
                spans.append(append(prefix + description, (len(prefix), len(prefix) + len(description))))
                positions.append(base_length + len(suffix) - 1)
                append("\n")
            # Inputs and the decision position are filled after all schemas are appended.
            branch = Branch({}, positions, -1, (0, base_length), instruction, spans)
            keys = [key for key, _ in pairs]
            compiled.append(Question(qid, question["type"], keys, [v for _, v in pairs], [branch],
                                     target_distribution(target, keys), not media))
        append("<|im_end|>\n<|im_start|>assistant\n")
        for index, question in enumerate(compiled):
            append(f"Question {index + 1} Decision:")
            question.branches[0].decision_position = base_length + len(suffix) - 1
            append("\n")
        length = base_length + len(suffix)
        if length > self.max_length:
            raise ValueError(f"Shared state: {length} tokens exceed max_length={self.max_length}; no truncation")
        inputs = dict(base)
        inputs["input_ids"] = torch.cat([base["input_ids"], base["input_ids"].new_tensor([suffix])], dim=1)
        inputs["attention_mask"] = torch.cat([base["attention_mask"],
                                             base["attention_mask"].new_ones((1, len(suffix)))], dim=1)
        if "mm_token_type_ids" in base:
            inputs["mm_token_type_ids"] = torch.cat([base["mm_token_type_ids"],
                                                    base["mm_token_type_ids"].new_zeros((1, len(suffix)))], dim=1)
        for question in compiled:
            question.branches[0].inputs = inputs
        if labeled_only:
            compiled = [question for question in compiled if question.target is not None]
        return CompiledState(compiled, length, length, media, inputs)
