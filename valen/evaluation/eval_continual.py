"""Evaluate t stage checkpoints with t model loads per rank, sharing each model across tasks.

Launch directly for one GPU or via torchrun for multiple GPUs. This module never
trains models. Dataset paths come from sequence.json; weights come from --run.
"""
import argparse
import gc
import json
from pathlib import Path

from valen.configuration import flatten_config
from evaluation.vqa_cl.calibration import (
    CalibrationTracker, digest, expected_questions, validate_options, write_json,
)


def load_plan(sequence, run):
    sequence, run = Path(sequence).resolve(), Path(run).resolve()
    manifest = json.loads(sequence.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if manifest.get("format") != "coin_decision_v1" or not tasks:
        raise ValueError("Expected a native continual sequence.json")
    names = [task["name"] for task in tasks]
    if (any(not isinstance(name, str) or not name or name in {".", ".."}
            or Path(name).name != name for name in names) or len(set(names)) != len(names)):
        raise ValueError("Task names must be unique directory names")
    training_plan = json.loads((run / "sequence_run.json").read_text(encoding="utf-8"))
    sequence_hash = digest(sequence)
    if training_plan.get("sequence_sha256") != sequence_hash or training_plan.get("tasks") != names:
        raise ValueError("Sequence differs from the training run")
    stages, datasets = [], []
    for i, task in enumerate(tasks):
        data = (sequence.parent / task["test"]).resolve()
        if digest(data) != task["test_sha256"]:
            raise ValueError(f"Prepared evaluation data changed: {data}")
        datasets.append({"path": str(data), "sha256": task["test_sha256"]})
        stage = run / f"{i:02d}_{task['name']}"
        config = flatten_config(json.loads((stage / "stage_config.json").read_text(encoding="utf-8")))
        checkpoint = stage / ("consolidated" if config.get("finetuning_type") == "cl_moe" else "latest")
        marker = json.loads((stage / "stage_complete.json").read_text(encoding="utf-8"))
        sha = digest(checkpoint / "checkpoint.pt")
        if marker.get("checkpoint_sha256") != sha:
            raise ValueError(f"Completed stage checkpoint changed: {checkpoint}")
        stages.append({"name": task["name"], "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
                       "config_sha256": digest(checkpoint / "config.json")})
    return {"sequence": str(sequence), "sequence_sha256": sequence_hash, "source_run": str(run),
            "tasks": names, "datasets": datasets, "stages": stages}


def summarize_matrix(matrix):
    """Match valen.continual's AP/AF/BWT definitions and expose loss matrices."""
    n = len(matrix)
    def values(key):
        return [[cell.get(key) if cell is not None else None for cell in row] for row in matrix]
    accuracy = values("accuracy")
    last = accuracy[-1]
    complete = all(value is not None for value in last)
    result = {"metrics_matrix": matrix, "accuracy_matrix": accuracy,
              "nll_matrix": values("nll"), "brier_matrix": values("brier"),
              "average_performance": sum(last) / n if complete else None,
              "completed_cells": sum(cell is not None for row in matrix for cell in row),
              "expected_cells": n * (n + 1) // 2,
              "average_performance_by_stage": [sum(row[:i + 1]) / (i + 1)
                  if all(v is not None for v in row[:i + 1]) else None for i, row in enumerate(accuracy)]}
    if n > 1 and complete and all(accuracy[i][i] is not None for i in range(n - 1)):
        bwt = sum(last[i] - accuracy[i][i] for i in range(n - 1)) / (n - 1)
        result.update(backward_transfer=bwt, average_forgetting=-bwt)
        if all(accuracy[t][i] is not None for i in range(n - 1) for t in range(i, n - 1)):
            result["max_past_forgetting"] = sum(
                max(accuracy[t][i] for t in range(i, n - 1)) - last[i] for i in range(n - 1)) / (n - 1)
    return result


def sealed_report(destination, checkpoint_sha, data_sha):
    try:
        marker = json.loads((destination / "evaluation_complete.json").read_text())
        expected = {"checkpoint_sha256": checkpoint_sha, "data_sha256": data_sha,
                    "metrics_sha256": digest(destination / "metrics.json"),
                    "predictions_sha256": digest(destination / "predictions.jsonl")}
        report = json.loads((destination / "metrics.json").read_text())
        if (marker == expected and report.get("checkpoint_sha256") == checkpoint_sha
                and report.get("data_sha256") == data_sha and "overall" in report.get("metrics", {})):
            return report
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def primary_call(context, operation):
    """Propagate primary-only filesystem/analysis errors to every rank."""
    result = None
    if context.primary:
        try:
            result = {"value": operation()}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
    result = context.gather(result)[0]
    if "error" in result:
        raise ValueError(result["error"])
    return result["value"]


def evaluate_sequence(plan, output, context, load_session, evaluate_dataset, release_memory,
                      *, resume=False, attn_implementation=None, calibration=True, bins=15, threshold=.8):
    """Stage-outer loop. Worker functions are injectable to test load counts."""
    output = Path(output).resolve()
    if output == Path(plan["source_run"]):
        raise ValueError("Use a separate evaluation output directory, not the training run root")
    validate_options(bins, threshold)
    identity = {**plan, "world_size": context.world_size, "attn_implementation": attn_implementation,
                "calibration": calibration, "bins": bins, "high_confidence_threshold": threshold}

    def prepare_output():
        if (output / "evaluation_run.json").exists():
            if not resume or json.loads((output / "evaluation_run.json").read_text()) != identity:
                raise ValueError("Evaluation output exists or inputs changed; use --resume with identical inputs")
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Nonempty evaluation output: {output}")
        if calibration:
            # Fail before any model loading if labels cannot support top-label ECE.
            for data in plan["datasets"]:
                expected_questions(data["path"])
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "evaluation_run.json", identity)
        # Analysis provenance, without modifying the original training artifacts.
        write_json(output / "sequence_run.json", {"tasks": plan["tasks"],
            "sequence_sha256": plan["sequence_sha256"], "source_run": plan["source_run"]})
        for i, stage in enumerate(plan["stages"]):
            folder = output / f"{i:02d}_{stage['name']}"
            folder.mkdir(exist_ok=True)
            write_json(folder / "stage_complete.json", {"checkpoint_sha256": stage["checkpoint_sha256"],
                                                       "source_checkpoint": stage["checkpoint"]})
    primary_call(context, prepare_output)
    tracker = None
    def make_tracker():
        nonlocal tracker
        if calibration:
            tracker = CalibrationTracker(plan["sequence"], output, bins, threshold, generic=True)
    primary_call(context, make_tracker)
    matrix = [[None] * len(plan["tasks"]) for _ in plan["tasks"]]
    loaded = 0
    for i, stage in enumerate(plan["stages"]):
        destinations = [output / f"{i:02d}_{stage['name']}" / "eval" / name for name in plan["tasks"][:i + 1]]
        reports = primary_call(context, lambda: [sealed_report(path, stage["checkpoint_sha256"],
            plan["datasets"][j]["sha256"]) if resume else None for j, path in enumerate(destinations)])
        session = None
        try:
            if any(report is None for report in reports):
                # Verify input weights/config again before loading, even on resume.
                if (digest(Path(stage["checkpoint"]) / "checkpoint.pt") != stage["checkpoint_sha256"]
                        or digest(Path(stage["checkpoint"]) / "config.json") != stage["config_sha256"]):
                    raise ValueError(f"Checkpoint changed during evaluation: {stage['checkpoint']}")
                session = load_session(stage["checkpoint"], context, attn_implementation)
                loaded += 1
                # print(json.dumps({"event": "continual_model_loaded", "rank": context.rank,
                #     "stage": i, "task": stage["name"], "model_loads_this_run": loaded}), flush=True)
            for j, destination in enumerate(destinations):
                if reports[j] is None:
                    primary_call(context, lambda: (destination / "evaluation_complete.json").unlink(missing_ok=True))
                    evaluate_dataset(session, plan["datasets"][j]["path"], destination)

                    def seal():
                        report = json.loads((destination / "metrics.json").read_text())
                        if (report["checkpoint_sha256"] != stage["checkpoint_sha256"]
                                or report["data_sha256"] != plan["datasets"][j]["sha256"]):
                            raise ValueError("Evaluation checkpoint/data mismatch")
                        write_json(destination / "evaluation_complete.json", {
                            "checkpoint_sha256": stage["checkpoint_sha256"],
                            "data_sha256": plan["datasets"][j]["sha256"],
                            "metrics_sha256": digest(destination / "metrics.json"),
                            "predictions_sha256": digest(destination / "predictions.jsonl")})
                        return report
                    reports[j] = primary_call(context, seal)
                matrix[i][j] = reports[j]["metrics"]["overall"]

                def summarize_cell():
                    if tracker is not None:
                        tracker.observe(i, j, destination, reports[j])
                    write_json(output / "continual_metrics.json", {"tasks": plan["tasks"],
                        "sequence_sha256": plan["sequence_sha256"], "source_run": plan["source_run"],
                        "model_loads_per_rank_this_run": loaded, **summarize_matrix(matrix)})
                primary_call(context, summarize_cell)
        finally:
            # A previous stage must not remain on GPU while the next is constructed.
            del session
            release_memory()
        context.barrier()
    return matrix


def run(sequence, run_dir, output=None, device="cuda", attn_implementation=None,
        resume=False, calibration=True, bins=15, threshold=.8):
    import torch
    from valen.training.distributed import initialize, close
    from .evaluate import EvaluationSession, evaluate_loaded
    context = initialize(device)
    try:
        plan = primary_call(context, lambda: load_plan(sequence, run_dir))
        def release_memory():
            gc.collect()
            if context.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return evaluate_sequence(plan, output or Path(run_dir) / "continual_evaluation", context,
            EvaluationSession, evaluate_loaded, release_memory, resume=resume,
            attn_implementation=attn_implementation, calibration=calibration, bins=bins, threshold=threshold)
    finally:
        close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--run", required=True, help="Training output containing 00_TASK, 01_TASK, ... checkpoints")
    parser.add_argument("--output", help="Separate evaluation directory; defaults to RUN/continual_evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip hard-label ECE analysis (needed for soft targets)")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--high-confidence-threshold", type=float, default=.8)
    args = parser.parse_args()
    run(args.sequence, args.run, args.output, args.device, args.attn_implementation, args.resume,
        not args.skip_calibration, args.bins, args.high_confidence_threshold)


if __name__ == "__main__":
    main()
