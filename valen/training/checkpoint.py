"""Store trainable weights, optimizer and RNG; frozen base weights stay pinned."""
import json
import random
from pathlib import Path
import torch
from valen.modeling.manifest import read_model_manifest, manifests_match


def file_sha256(path):
    # 全参数 ckpt 可能很大，分块校验。 / Hash large full-training checkpoints in chunks.
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rank_state(progress, rng):
    return {"progress": progress, "rng": rng.getstate(), "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng_current": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def save_checkpoint(path, model, optimizer, config, progress, rng, rank_states=None, training_state=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    delta = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}
    from valen.modeling.qwen.cl_moe import checkpoint_buffers
    buffers = {name: value.detach().cpu().clone() for name, value in checkpoint_buffers(model).items()}
    manifest = getattr(model, "base_manifest", None) or read_model_manifest(config)
    payload = {"weights": delta, "optimizer": optimizer.state_dict(), "config": config,
               **capture_rank_state(progress, rng),
               "base_manifest": manifest}
    if rank_states is not None:
        payload["distributed"] = {"world_size": len(rank_states), "rank_states": rank_states}
    if training_state is not None:
        payload["training_state"] = training_state
    if buffers:
        payload["adaptation_buffers"] = buffers
    temp = path / "checkpoint.tmp"
    torch.save(payload, temp)
    temp.replace(path / "checkpoint.pt")
    (path / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path, model, optimizer=None, rng=None, strict=True, rank=0, world_size=1):
    # model 必须已构建并加载对应基座；本函数将训练时保存的参数覆盖到该模型上。
    # checkpoint 还包含优化器、进度和随机状态，因此读取完整 payload，而不只是权重字典。
    # 先映射到 CPU，mmap 避免一次性读入所有张量存储；只加载本项目生成的可信 checkpoint。
    payload = torch.load(Path(path) / "checkpoint.pt", map_location="cpu", weights_only=False, mmap=True)

    # 传入 optimizer 或 rng 表示需要恢复训练状态。
    # CL-MoE 的 consolidated 权重经过专家融合，与融合前的优化器状态不再对应，
    # 因而只能用于评测或初始化新阶段；断点续训应使用融合前的 latest。
    if (optimizer is not None or rng is not None) and (payload.get("training_state") or {}).get("cl_moe_consolidated", False):
        raise ValueError("Consolidated CL-MoE weights require --initialize; resume the raw latest checkpoint instead")
    if optimizer is not None or rng is not None:
        # 恢复训练必须保持原进程数，确保每个 rank 的数据分片、游标和随机状态仍然对应。
        # 仅加载权重做评测或初始化时，不受这个 world_size 检查限制。
        saved_world_size = payload.get("distributed", {}).get("world_size", 1)
        if saved_world_size != world_size:
            raise ValueError(f"Resume requires world_size={saved_world_size}, got {world_size}; use --initialize to change GPU count")
        if "distributed" in payload:
            # 将当前 rank 自己的 progress 和随机状态覆盖到 payload 顶层，供后续恢复。
            payload.update(payload["distributed"]["rank_states"][rank])

    # 保存时只记录 requires_grad=True 的参数；冻结的基座参数由 build_model 加载。
    # 因此，即使现在用于评测，加载前也要保留与 checkpoint 匹配的参数可训练标记。
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    actual = set(payload["weights"])
    # strict=True：要求当前可训练参数名与保存的参数名完全一致。
    if strict and expected != actual:
        raise ValueError(f"Trainable checkpoint mismatch: missing={expected-actual}, unexpected={actual-expected}")
    # strict=False：允许新阶段增加可训练参数，但不能遗漏或冻结旧阶段已更新的参数。
    if not strict and actual - expected:
        raise ValueError("Initialization would freeze previously changed parameters; use a stage with a superset of trainable parameters")

    # 校验基座来源信息，避免将适配权重叠加到不匹配的基座版本上。
    base = payload.get("base_manifest")
    if base:
        current = getattr(model, "base_manifest", None)
        if not manifests_match(base, current):
            raise ValueError("Base model revision differs from checkpoint")

    # 这里的 strict=False 是 PyTorch 的 state_dict 加载选项，与上面的参数名校验分开。
    # checkpoint 不含冻结基座的完整参数，所以允许 missing_keys；未知参数仍然报错。
    result = model.load_state_dict(payload["weights"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unknown checkpoint weights: {result.unexpected_keys}")

    # CL-MoE 的任务原型、任务路由等 buffer 不属于可训练参数，需单独检查并恢复。
    # 普通模型在这里通常得到两个空字典；两边 buffer 名必须完全一致。
    from valen.modeling.qwen.cl_moe import checkpoint_buffers
    expected_buffers = checkpoint_buffers(model)
    saved_buffers = payload.get("adaptation_buffers", {})
    if set(saved_buffers) != set(expected_buffers):
        raise ValueError("CL-MoE checkpoint adaptation buffers mismatch")
    if saved_buffers:
        model.load_state_dict(saved_buffers, strict=False)

    # 只有调用方传入优化器才恢复其状态，例如 AdamW 的动量和步数。
    # 评测或初始化新实验时不传 optimizer，便不会恢复旧优化器状态。
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])

    # rng 是训练流程使用的独立 random.Random 对象；同时恢复 Python 全局、
    # PyTorch CPU 和 CUDA 的随机状态，使后续数据顺序与随机操作衔接断点。
    if rng is not None:
        rng.setstate(payload["rng"])
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available():
            if payload.get("cuda_rng_current") is not None:
                # 当前格式：各 rank 恢复自己当前 CUDA 设备的随机状态。
                torch.cuda.set_rng_state(payload["cuda_rng_current"])
            elif payload.get("cuda_rng"):
                # 兼容旧格式：一次恢复保存的所有 CUDA 设备随机状态。
                torch.cuda.set_rng_state_all(payload["cuda_rng"])

    # 将完整 payload 返回给调用方；例如 runner 恢复 progress，评测读取训练 step/epoch。
    return payload
