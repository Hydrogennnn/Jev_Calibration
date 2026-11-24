"""CL-MoE LoRA experts and dual routing, adapted to Qwen decision inputs.

Reference: Huai et al., CVPR 2025, arXiv:2503.00413, equations 3, 5–12.
The total rank is divided between experts; alpha / total_rank is preserved.
No task ID, filesystem access, or routing-statistic mutation during training.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn


def normalize_options(config):
    if config.get("finetuning_type") != "cl_moe":
        if config.get("cl_moe"):
            raise ValueError("cl_moe options require finetuning_type=cl_moe")
        return config
    if config.get("architecture", "qwen") != "qwen":
        raise ValueError("CL-MoE requires architecture=qwen")
    if config.get("stage", "joint") not in {"text", "joint", "vision_top"}:
        raise ValueError("CL-MoE requires text, joint or vision_top stage")
    if config.get("method", "sft") != "sft":
        raise ValueError("CL-MoE currently supports SFT only")
    defaults = dict(experts=4, top_k=2, max_tasks=8, beta=0.5, gamma=0.7,
                    calibration_limit=0)
    supplied = config.get("cl_moe", {})
    if not isinstance(supplied, dict) or set(supplied) - set(defaults):
        raise ValueError("Unknown CL-MoE options")
    options = dict(defaults, **supplied)
    for name in ("experts", "top_k", "max_tasks"):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f"cl_moe.{name} must be a positive integer")
    if options["top_k"] > options["experts"]:
        raise ValueError("cl_moe.top_k cannot exceed experts")
    for name in ("beta", "gamma"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"cl_moe.{name} must be in [0, 1]")
    if options["gamma"] <= 0.5:
        raise ValueError("cl_moe.gamma must be greater than 0.5")
    limit = options["calibration_limit"]
    if type(limit) is not int or limit < 0:
        raise ValueError("cl_moe.calibration_limit must be a nonnegative integer")
    rank = config.setdefault("lora_rank", 32)
    alpha = config.setdefault("lora_alpha", 64)
    if type(rank) is not int or rank < options["experts"] or rank % options["experts"]:
        raise ValueError("lora_rank must be positive and divisible by cl_moe.experts")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("lora_alpha must be finite and positive")
    task = config.setdefault("cl_moe_task", 0)
    if type(task) is not int or not 0 <= task < options["max_tasks"]:
        raise ValueError("cl_moe_task must be in [0, max_tasks)")
    config["cl_moe"] = options
    return config


class CLMoELinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, experts, max_tasks):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("CL-MoE supports nn.Linear targets")
        self.base_layer = base_layer.requires_grad_(False)
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.scaling = alpha / rank
        self.experts = experts
        device = base_layer.weight.device
        self.lora_A = nn.ModuleList(nn.Linear(self.in_features, rank // experts, bias=False,
                                            device=device, dtype=torch.float32) for _ in range(experts))
        self.lora_B = nn.ModuleList(nn.Linear(rank // experts, self.out_features, bias=False,
                                            device=device, dtype=torch.float32) for _ in range(experts))
        self.lora_router = nn.Linear(self.in_features, experts, bias=False, device=device, dtype=torch.float32)
        for a, b in zip(self.lora_A, self.lora_B):
            nn.init.normal_(a.weight, std=0.01)
            nn.init.zeros_(b.weight)
        self.register_buffer("task_routes", torch.zeros(max_tasks, experts, device=device))
        self.register_buffer("used_experts", torch.zeros(experts, dtype=torch.bool, device=device))
        self.register_buffer("route_sum", torch.zeros(experts, dtype=torch.float64, device=device), persistent=False)
        self.register_buffer("route_count", torch.zeros((), dtype=torch.float64, device=device), persistent=False)
        self.routing_weights = None
        self.routing_mask = None
        self.collect_routes = False
        self.beta = 0.5

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        result = self.base_layer(x)
        adapted = x.to(self.lora_router.weight.dtype)
        probs = self.lora_router(adapted).float().softmax(-1)
        if self.collect_routes:
            if probs.ndim != 3 or self.routing_mask is None or probs.shape[:2] != self.routing_mask.shape:
                raise ValueError("CL-MoE calibration requires [batch, sequence, hidden] Linear inputs")
            mask = self.routing_mask.to(probs.dtype)
            means = (probs.detach() * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
            self.route_sum.add_(means.double().sum(0))
            self.route_count.add_(means.shape[0])
        if self.routing_weights is not None:
            if probs.ndim != 3 or probs.shape[0] != self.routing_weights.shape[0]:
                raise ValueError("CL-MoE dual routing requires batched sequence inputs")
            probs = self.beta * probs + (1 - self.beta) * self.routing_weights[:, None, :]
        update = None
        for i, (a, b) in enumerate(zip(self.lora_A, self.lora_B)):
            value = b(a(adapted)) * probs[..., i:i + 1]
            update = value if update is None else update + value
        return result + (update * self.scaling).to(result.dtype)


def layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, CLMoELinear)]


def inject(language_model, config):
    targets = [(name, module) for name, module in language_model.named_modules() if isinstance(module, nn.Linear)]
    if not targets:
        raise ValueError("No language Linear layers found for CL-MoE")
    for name, base in targets:
        parent_name, _, child = name.rpartition(".")
        parent = language_model.get_submodule(parent_name) if parent_name else language_model
        setattr(parent, child, CLMoELinear(base, config["lora_rank"], config["lora_alpha"],
                config["cl_moe"]["experts"], config["cl_moe"]["max_tasks"]))
    return [name for name, _ in targets]


class CLMoEState(nn.Module):
    """Stable task indexing using mean frozen *text input embeddings*.

    Qwen has no CLS encoder readout. This explicit adaptation avoids a second
    backbone pass and prevents prototypes drifting as language adapters learn.
    Prototypes include the textual state, instructions and candidate descriptions.
    """
    def __init__(self, hidden_size, max_tasks, beta):
        super().__init__()
        self.register_buffer("centers", torch.zeros(max_tasks, hidden_size))
        self.register_buffer("completed_tasks", torch.zeros((), dtype=torch.long))
        self.register_buffer("center_sum", torch.zeros(hidden_size, dtype=torch.float64), persistent=False)
        self.register_buffer("center_count", torch.zeros((), dtype=torch.float64), persistent=False)
        self.beta = beta
        self.calibrating = False

    @contextmanager
    def route(self, model, inputs):
        if model.training and not self.calibrating:
            # Training/checkpoint recomputation is pure instance routing.
            yield
            return
        modules = layers(model.backbone.language_model)
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids)).bool()
        if mask.ndim != 2:
            raise ValueError("CL-MoE expects a two-dimensional padding mask")
        text_mask = mask.clone()
        for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            token = getattr(model.backbone.config, key, None)
            if token is not None:
                text_mask &= ids != token
        with torch.no_grad():
            features = model.backbone.get_input_embeddings()(ids).float()
            features = (features * text_mask[..., None]).sum(1) / text_mask.sum(1).clamp_min(1)[:, None]
            count = int(self.completed_tasks)
            index = None
            if self.calibrating:
                self.center_sum.add_(features.double().sum(0))
                self.center_count.add_(features.shape[0])
            elif count:
                index = torch.cdist(features, self.centers[:count].float()).argmin(-1)
        try:
            for _, layer in modules:
                layer.beta = self.beta
                layer.routing_mask = mask
                layer.collect_routes = self.calibrating
                layer.routing_weights = layer.task_routes[index] if index is not None else None
            yield
        finally:
            for _, layer in modules:
                layer.routing_weights = layer.routing_mask = None
                layer.collect_routes = False


def checkpoint_buffers(model):
    """Only adaptation state, never the pretrained backbone's buffers."""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, (CLMoELinear, CLMoEState)):
            for key, value in module.named_buffers(recurse=False):
                if key not in module._non_persistent_buffers_set:
                    result[f"{name}.{key}"] = value
    return result
