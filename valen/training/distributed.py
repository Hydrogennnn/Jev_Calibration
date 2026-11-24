"""Synchronous data parallelism for unequal numbers of question backwards.

Gradients are summed only after local accumulation, so an idle rank and a
text-only rank participate in the same collectives as a visual/Score rank.
"""
from dataclasses import dataclass
from datetime import timedelta
import os
import random
import torch
import torch.distributed as dist


@dataclass
class Distributed:
    rank: int = 0
    world_size: int = 1
    device: str = "cpu"

    @property
    def primary(self):
        return self.rank == 0

    def sum(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()


def initialize(device="cuda"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if str(device).startswith("cuda"):
        index = int(os.environ["LOCAL_RANK"]) if world_size > 1 else torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        torch.cuda.set_device(index)
        device = f"cuda:{index}"
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl" if str(device).startswith("cuda") else "gloo",
                                    timeout=timedelta(minutes=10))
        rank, world_size = dist.get_rank(), dist.get_world_size()
    return Distributed(rank, world_size, str(device))


def close():
    if dist.is_initialized():
        dist.destroy_process_group()


def epoch_shard(size, seed, epoch, rank, world_size):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    # No padding or dropped tail: each state belongs to exactly one rank.
    return order[rank::world_size]


@torch.no_grad()
def broadcast_trainable(model, context):
    if context.world_size > 1:
        for parameter in model.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter, src=0)


@torch.no_grad()
def synchronize_gradients(model, context, bucket_bytes=16 * 1024 * 1024):
    """各卡完成本地梯度累积后，对可训练参数的梯度求全局和。

    所有 rank 必须以相同的参数顺序调用本函数，即使某张卡本轮没有
    执行 backward。同步及梯度赋值无需构建计算图，因此使用 no_grad。
    """
    if context.world_size == 1:
        return

    # 冻结参数不参与同步。先标记本卡哪些参数有梯度（有梯度但值全零也算）。
    parameters = [p for p in model.parameters() if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=context.device)
    # 逐元素取所有卡的最大值，相当于逻辑 OR：任意卡有梯度，标记就是 1。
    # 这样各卡会得到相同的 active 列表，后续通信的次数、顺序和形状才能一致。
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    # 所有卡都未使用的参数保留 grad=None，使 AdamW 跳过这些参数的更新
    # （包括权重衰减）；不能把它们一律替换成零梯度。
    active = [p for p, flag in zip(parameters, used.tolist()) if flag]

    # 将多个参数的梯度合并成通信桶，减少逐个参数 all_reduce 的调用开销。
    # 每桶使用相同 dtype，目标大小为 16 MiB；单个参数不拆分，因此大参数
    # 可以独占一个超过 bucket_bytes 的桶。这里假设参数和梯度 dtype 一致。
    buckets, bucket, size = [], [], 0
    for p in active:
        nbytes = p.numel() * p.element_size()
        if bucket and (size + nbytes > bucket_bytes or p.dtype != bucket[0].dtype):
            buckets.append(bucket)
            bucket, size = [], 0
        bucket.append(p)
        size += nbytes
    if bucket:
        buckets.append(bucket)

    for bucket in buckets:
        # 展平并拼接本桶梯度。若参数仅在其他卡上使用，本卡以零贡献参与求和，
        # 不能跳过该参数，否则各卡通信张量的形状或内容对应关系会不一致。
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        # 原地求和，并将结果同步到每张卡。runner 已按全局样本数或问题数
        # 对 loss 归一化，因此这里不再除以 world_size，避免重复缩小梯度。
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)

        # 按拼接顺序切片并恢复参数形状，让 optimizer.step() 使用同步后的梯度。
        # 这些 grad 是 flat 的视图，共享桶的存储；原先无本地梯度的参数也会
        # 获得全局梯度，从而与其他卡执行一致的更新。
        offset = 0
        for p in bucket:
            p.grad = flat[offset:offset + p.numel()].view_as(p)
            offset += p.numel()
