from types import SimpleNamespace

import pytest
import torch
from torch import nn

from valen.modeling.qwen.cl_moe import CLMoELinear, CLMoEState, normalize_options
from valen.training.cl_moe import ContinualMoE
from valen.training.distributed import Distributed


def test_zero_initialization_gradients_and_masked_calibration():
    torch.manual_seed(3)
    base = nn.Linear(3, 2)
    layer = CLMoELinear(base, rank=8, alpha=16, experts=4, max_tasks=3)
    x = torch.randn(2, 5, 3)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    layer(x).sum().backward()
    assert base.weight.grad is None
    assert sum(b.weight.grad.abs().sum() for b in layer.lora_B) > 0
    layer.collect_routes = True
    layer.routing_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    layer(x)
    probs = layer.lora_router(x).softmax(-1)
    expected = probs[0, :2].mean(0) + probs[1, :3].mean(0)
    torch.testing.assert_close(layer.route_sum.float(), expected)
    assert layer.route_count == 2


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.language_model = CLMoELinear(nn.Linear(3, 2), 8, 16, 4, 3)
        self.cl_moe_state = CLMoEState(3, 3, .5)


def test_momentum_uses_current_minus_entire_history_and_first_task_unmixed():
    model = Tiny()
    layer = model.backbone.language_model
    options = normalize_options(dict(finetuning_type="cl_moe", lora_rank=8, cl_moe_task=1,
                                     cl_moe=dict(max_tasks=3)))
    model.cl_moe_state.completed_tasks.fill_(1)
    layer.used_experts[:2] = True
    with torch.no_grad():
        for blocks in (layer.lora_A, layer.lora_B):
            for block in blocks:
                block.weight.fill_(10.)
    continual = ContinualMoE(model, options, {"training_state": {"cl_moe_consolidated": True}})
    with torch.no_grad():
        for blocks in (layer.lora_A, layer.lora_B):
            for block in blocks:
                block.weight.fill_(20.)

    class Backend:
        def inference_units(self, model, compiled):
            layer.route_sum += torch.tensor([.05, .35, .5, .1])
            layer.route_count += 1
            model.cl_moe_state.center_sum += torch.tensor([1., 2., 3.])
            model.cl_moe_state.center_count += 1
            yield []

    compiler = SimpleNamespace(compile=lambda *a, **kw: None)
    report = continual.consolidate(model, compiler, Backend(), [{}], Distributed())
    for i, value in enumerate([13., 13., 17., 13.]):
        torch.testing.assert_close(layer.lora_A[i].weight, torch.full_like(layer.lora_A[i].weight, value))
        torch.testing.assert_close(layer.lora_B[i].weight, torch.full_like(layer.lora_B[i].weight, value))
    assert layer.used_experts.tolist() == [True, True, True, False]
    assert model.cl_moe_state.completed_tasks == 2
    assert report["layers"]["backbone.language_model"]["new_specific"] == [2]

    fresh = Tiny()
    options["cl_moe_task"] = 0
    first = ContinualMoE(fresh, options)
    with torch.no_grad():
        for b in fresh.backbone.language_model.lora_B:
            b.weight.fill_(7.)
    # Use the calibration backend with the new module.
    layer = fresh.backbone.language_model
    first.consolidate(fresh, compiler, Backend(), [{}], Distributed())
    assert torch.all(layer.lora_B[0].weight == 7)


def test_inference_nearest_prototype_dual_route_without_task_label():
    model = Tiny().eval()
    embedding = nn.Embedding(6, 3)
    with torch.no_grad():
        embedding.weight[1].fill_(1)
        embedding.weight[2].fill_(-1)
        embedding.weight[3].fill_(100)  # image placeholder must be ignored
    model.backbone.get_input_embeddings = lambda: embedding
    model.backbone.config = SimpleNamespace(image_token_id=3)
    model.cl_moe_state.completed_tasks.fill_(2)
    model.cl_moe_state.centers[0].fill_(1)
    model.cl_moe_state.centers[1].fill_(-1)
    layer = model.backbone.language_model
    layer.task_routes[0] = torch.tensor([1., 0., 0., 0.])
    layer.task_routes[1] = torch.tensor([0., 0., 1., 0.])
    inputs = dict(input_ids=torch.tensor([[1, 3, 0], [2, 2, 3]]),
                  attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]))
    with model.cl_moe_state.route(model, inputs):
        torch.testing.assert_close(layer.routing_weights, layer.task_routes[:2])
        with torch.no_grad():
            for i, (a, b) in enumerate(zip(layer.lora_A, layer.lora_B)):
                a.weight.fill_(1)
                b.weight.fill_(i + 1)
        x = torch.ones(2, 3, 3)
        probs = layer.lora_router(x).softmax(-1) * .5 + layer.task_routes[:2, None, :] * .5
        expected = layer.base_layer(x) + sum(b(a(x)) * probs[..., i:i+1]
                   for i, (a, b) in enumerate(zip(layer.lora_A, layer.lora_B))) * layer.scaling
        torch.testing.assert_close(layer(x), expected)
    assert layer.routing_weights is None and layer.routing_mask is None
    assert layer.route_count == 0


@pytest.mark.parametrize("change", [dict(lora_rank=3), dict(cl_moe={"top_k": 5}),
    dict(cl_moe={"gamma": .5}), dict(cl_moe={"beta": float("nan")}), dict(stage="warmup"),
    dict(method="rlcd"), dict(architecture="dual_encoder"), dict(cl_moe_task=-1)])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        normalize_options(dict(finetuning_type="cl_moe", **change))
