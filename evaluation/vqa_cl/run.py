"""Run the CL4VQA ten-task sequence using Valen's existing continual trainer."""
import argparse
import json
from pathlib import Path

from valen.configuration import flatten_config
from .protocol import BENCHMARK, task_order
from .calibration import CalibrationTracker, validate_options


def run(config, sequence, output, gpus=1, resume=False, dry_run=False, evaluate=True,
        *, ece_bins=15, high_confidence_threshold=.8):
    # Keep the lightweight prepare/download commands independent of torch.
    from valen.continual import run_sequence
    from valen.training.schedule import per_rank_batch, validate_schedule
    validate_options(ece_bins, high_confidence_threshold)
    manifest = json.loads(Path(sequence).read_text())
    if manifest.get("benchmark") != BENCHMARK:
        raise ValueError("Expected sequence.json from evaluation.vqa_cl.prepare, not the CoIN sequence")
    names = [task["name"] for task in manifest["tasks"]]
    if task_order(names) != names:
        raise ValueError("Sequence must contain canonical CL4VQA task names")
    options = flatten_config(json.loads(Path(config).read_text()))
    if options.get("architecture") != "qwen" or options.get("qwen_execution") != "shared_state":
        raise ValueError("This CL4VQA profile requires Qwen shared_state execution")
    if options.get("method") != "sft" or options.get("finetuning_type") not in {"cl_moe", "lora"}:
        raise ValueError("Use SFT with cl_moe or the sequential lora baseline")
    validate_schedule(options)
    if gpus < 1:
        raise ValueError("gpus must be positive")
    if options.get("global_batch_size"):
        per_rank_batch(options, gpus)
    print(json.dumps({"benchmark": BENCHMARK, "tasks": names, "smoke_limit": manifest["smoke_limit"],
                      "primitives": manifest["primitives"], "warning": manifest["warning"]}), flush=True)
    tracker = CalibrationTracker(sequence, output, ece_bins, high_confidence_threshold) if evaluate and not dry_run else None
    return run_sequence(config, sequence, output, gpus, resume, dry_run, evaluate,
                        on_evaluation=tracker.observe if tracker else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/qwen/vqa_cl_moe.json")
    parser.add_argument("--sequence", default="data/vqa-cl-decision/sequence.json")
    parser.add_argument("--output", default="output/vqa-cl-moe")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--high-confidence-threshold", type=float, default=.8)
    args = parser.parse_args()
    run(args.config, args.sequence, args.output, args.gpus, args.resume, args.dry_run, not args.skip_eval,
        ece_bins=args.ece_bins, high_confidence_threshold=args.high_confidence_threshold)


if __name__ == "__main__":
    main()
