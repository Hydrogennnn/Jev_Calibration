"""End-of-task route calibration and dynamic momentum consolidation.

Unlike the reference repository's wall-clock sampled text logs and params.py,
statistics are deterministic, padding-aware and reduced over all training ranks.
Expert sets follow paper Eq. 10: current-minus-history receives the larger NEW
weight. The history is the union of every preceding task, not just the last one.
"""
import torch

from valen.modeling.qwen.cl_moe import layers
from tqdm import tqdm

def expert_parameters(model):
    return {name: p for name, p in model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name}


class ContinualMoE:
    def __init__(self, model, config, payload=None, resuming=False):
        self.config = config
        self.task = config["cl_moe_task"]
        if int(model.cl_moe_state.completed_tasks) != self.task:
            raise ValueError("cl_moe_task must equal completed_tasks; initialize from the previous consolidated checkpoint")
        saved = (payload or {}).get("training_state") or {}
        if resuming:
            if "cl_moe" not in saved or saved.get("cl_moe_consolidated"):
                raise ValueError("Resume requires an unconsolidated CL-MoE latest checkpoint")
            self.previous = saved["cl_moe"]["previous_experts"]
            if saved["cl_moe"]["task"] != self.task:
                raise ValueError("CL-MoE resume task mismatch")
        else:
            if self.task and not saved.get("cl_moe_consolidated"):
                raise ValueError("The previous task must be consolidated before starting the next task")
            self.previous = ({n: p.detach().cpu().clone() for n, p in expert_parameters(model).items()}
                             if self.task else {})
        expected = expert_parameters(model) if self.task else {}
        if set(self.previous) != set(expected) or any(self.previous[n].shape != p.shape for n, p in expected.items()):
            raise ValueError("CL-MoE momentum anchor mismatch")

    def state_dict(self):
        return {"task": self.task, "previous_experts": self.previous}

    @torch.no_grad()
    def consolidate(self, model, compiler, backend, records, context):
        options = self.config["cl_moe"]
        limit = options["calibration_limit"] or len(records)
        count = min(limit, len(records))
        state = model.cl_moe_state
        state.center_sum.zero_()
        state.center_count.zero_()
        modules = layers(model)
        for _, layer in modules:
            layer.route_sum.zero_()
            layer.route_count.zero_()
        model.eval()
        state.calibrating = True
        try:
            for index in tqdm(
                range(context.rank, count, context.world_size),
                desc=f"Rank {context.rank} CL-MoE calibration",
                position=context.rank,
                leave=True,
                dynamic_ncols=True,
            ):
                compiled = compiler.compile(records[index], labeled_only=True)
                for _ in backend.inference_units(model, compiled):
                    pass
        finally:
            state.calibrating = False
        summed = torch.tensor(context.sum(state.center_sum.tolist() + [state.center_count.item()]),
                              device=state.centers.device, dtype=torch.float64)
        if summed[-1] <= 0:
            raise ValueError("No labeled inputs available for CL-MoE calibration")
        state.centers[self.task].copy_((summed[:-1] / summed[-1]).float())
        reports = {}
        for name, layer in modules:
            totals = torch.tensor(context.sum(layer.route_sum.tolist() + [layer.route_count.item()]),
                                  device=layer.task_routes.device, dtype=torch.float64)
            if totals[-1] <= 0:
                raise ValueError(f"No routing observations for {name}")
            weights = (totals[:-1] / totals[-1]).float()
            layer.task_routes[self.task].copy_(weights)
            # Stable index tie-breaking makes task sets reproducible.
            selected = torch.argsort(weights, descending=True, stable=True)[:options["top_k"]]
            current = torch.zeros_like(layer.used_experts)
            current[selected] = True
            specific = current & ~layer.used_experts
            old_weights = torch.full_like(weights, options["gamma"])
            old_weights[specific] = 1 - options["gamma"]
            if self.task:
                for i in range(layer.experts):
                    for block in ("lora_A", "lora_B"):
                        parameter = getattr(layer, block)[i].weight
                        anchor = self.previous[f"{name}.{block}.{i}.weight"].to(parameter)
                        parameter.mul_(1 - old_weights[i]).add_(anchor * old_weights[i])
            reports[name] = {"route": weights.tolist(), "selected": selected.tolist(),
                "new_specific": specific.nonzero().flatten().tolist(),
                "history": layer.used_experts.nonzero().flatten().tolist(),
                "old_weight": old_weights.tolist() if self.task else None,
                "observations": int(totals[-1])}
            layer.used_experts.logical_or_(current)
        state.completed_tasks.add_(1)
        return {"task": self.task, "calibration_records": count, "available_records": len(records),
                "prototype": "mean_frozen_text_input_embeddings", "layers": reports,
                "momentum": "paper_equations_10_12_on_lora_factors", "first_task_unmixed": self.task == 0}
