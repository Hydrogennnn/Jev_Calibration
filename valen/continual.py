"""Run a CoIN-Decision task sequence and evaluate every seen task after each stage."""
import argparse
import json
from pathlib import Path
import sys

from valen.configuration import flatten_config
from valen.processes import run_command
from valen.training.checkpoint import file_sha256


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize_matrix(matrix):
    """Values are fractions; AF uses the CL-MoE paper's diagonal definition."""
    if not matrix:
        return {}
    n = len(matrix)
    accuracy = [[cell.get("accuracy") if cell else None for cell in row] for row in matrix]
    last = accuracy[-1][:n]
    result = {"accuracy_matrix": accuracy, "metrics_matrix": matrix,
              "average_performance": sum(last) / n if all(v is not None for v in last) else None}
    if n > 1 and all(accuracy[i][i] is not None and last[i] is not None for i in range(n - 1)):
        bwt = sum(last[i] - accuracy[i][i] for i in range(n - 1)) / (n - 1)
        result.update(backward_transfer=bwt, average_forgetting=-bwt,
            max_past_forgetting=sum(max(accuracy[t][i] for t in range(i, n - 1)) - last[i]
                                    for i in range(n - 1)) / (n - 1))
    return result


def command(module, args, gpus):
    prefix = [sys.executable]
    if gpus > 1:
        prefix += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"]
    return prefix + ["-m", module] + list(map(str, args))


def evaluation_signature(destination, checkpoint_sha256, data_sha256):
    return {"checkpoint_sha256": checkpoint_sha256, "data_sha256": data_sha256,
            "metrics_sha256": file_sha256(destination / "metrics.json"),
            "predictions_sha256": file_sha256(destination / "predictions.jsonl")}


def completed_evaluation(destination, checkpoint_sha256, data_sha256):
    """Only reuse outputs sealed after the evaluator exits successfully."""
    try:
        marker = json.loads((destination / "evaluation_complete.json").read_text())
        if marker != evaluation_signature(destination, checkpoint_sha256, data_sha256):
            return None
        report = json.loads((destination / "metrics.json").read_text())
        if (report.get("checkpoint_sha256") == checkpoint_sha256 and
                report.get("data_sha256") == data_sha256 and "overall" in report.get("metrics", {})):
            return report
    except (OSError, ValueError):
        pass
    return None


def run_sequence(config_path, sequence_path, output, gpus=1, resume=False, dry_run=False, evaluate=True,
                 *, on_evaluation=None):
    config_path, sequence_path, output = map(lambda p: Path(p).resolve(), (config_path, sequence_path, output))
    config = flatten_config(json.loads(config_path.read_text(encoding="utf-8")))
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence.get("format") != "coin_decision_v1" or not sequence.get("tasks"):
        raise ValueError("Expected a CoIN-Decision sequence.json from evaluation.coin.prepare")
    cl_moe = config.get("finetuning_type") == "cl_moe"
    if cl_moe and config.get("cl_moe", {}).get("max_tasks", 8) < len(sequence["tasks"]):
        raise ValueError("cl_moe.max_tasks is smaller than the sequence")
    names = [task["name"] for task in sequence["tasks"]]
    if any(not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}
           for name in names) or len(set(names)) != len(names):
        raise ValueError("Task names must be unique, valid directory names")
    task_paths = []
    for task in sequence["tasks"]:
        paths = {}
        for split in ("train", "test"):
            path = (sequence_path.parent / task[split]).resolve()
            if file_sha256(path) != task[f"{split}_sha256"]:
                raise ValueError(f"Prepared {split} data changed for {task['name']}")
            paths[split] = path
        task_paths.append(paths)
    plan = {"config_sha256": file_sha256(config_path), "sequence_sha256": file_sha256(sequence_path),
            "gpus": gpus, "evaluate": evaluate, "tasks": [t["name"] for t in sequence["tasks"]]}
    plan_path = output / "sequence_run.json"
    if plan_path.exists():
        if not resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Output exists or resume config/sequence changed; use --resume with identical inputs")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    matrix = []
    previous = None
    for i, (task, paths) in enumerate(zip(sequence["tasks"], task_paths)):
        name = task["name"]
        folder = output / f"{i:02d}_{name}"
        folder.mkdir(exist_ok=True)
        stage_config = dict(config, data=str(paths["train"]), output=str(folder))
        if cl_moe:
            stage_config["cl_moe_task"] = i
        stage_path = folder / "stage_config.json"
        write_json(stage_path, stage_config)
        checkpoint = folder / ("consolidated" if cl_moe else "latest")
        marker = folder / "stage_complete.json"
        if marker.exists():
            if json.loads(marker.read_text())["checkpoint_sha256"] != file_sha256(checkpoint / "checkpoint.pt"):
                raise ValueError("Completed stage checkpoint changed")
        else:
            args = ["--config", stage_path]
            raw = folder / "latest"
            if resume and (raw / "checkpoint.pt").exists():
                args += ["--resume", raw]
            elif previous is not None:
                args += ["--initialize", previous]
            cmd = command("valen.train", args, gpus)
            print(json.dumps({"stage": name, "command": cmd}), flush=True)
            if not dry_run:
                run_command(cmd)
                write_json(marker, {"checkpoint_sha256": file_sha256(checkpoint / "checkpoint.pt")})
        checkpoint_sha256 = file_sha256(checkpoint / "checkpoint.pt") if not dry_run else None
        row = []
        if evaluate:
            for j in range(i + 1):
                destination = folder / "eval" / sequence["tasks"][j]["name"]
                report_path = destination / "metrics.json"
                data_sha256 = sequence["tasks"][j]["test_sha256"]
                report = completed_evaluation(destination, checkpoint_sha256, data_sha256) if not dry_run else None
                if report is None:
                    args = ["--checkpoint", checkpoint, "--data", task_paths[j]["test"],
                            "--output", destination, "--device", config.get("device", "cuda")]
                    if config.get("attn_implementation"):
                        args += ["--attn-implementation", config["attn_implementation"]]
                    cmd = command("valen.evaluate", args, gpus)
                    print(json.dumps({"evaluate": j, "command": cmd}), flush=True)
                    if not dry_run:
                        completion = destination / "evaluation_complete.json"
                        completion.unlink(missing_ok=True)
                        run_command(cmd)
                        report = json.loads(report_path.read_text())
                        if (report.get("checkpoint_sha256") != checkpoint_sha256 or
                                report.get("data_sha256") != data_sha256):
                            raise ValueError("Evaluation checkpoint/data changed while running")
                        write_json(completion, evaluation_signature(destination, checkpoint_sha256, data_sha256))
                if not dry_run:
                    if on_evaluation is not None:
                        on_evaluation(i, j, destination, report)
                    row.append(report["metrics"]["overall"])
            if not dry_run:
                matrix.append(row + [None] * (len(sequence["tasks"]) - len(row)))
                write_json(output / "continual_metrics.json", dict(tasks=plan["tasks"], **summarize_matrix(matrix)))
        previous = checkpoint
    return matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Write stage configs and print commands without loading models")
    p.add_argument("--skip-eval", action="store_true")

    a = p.parse_args()
    run_sequence(a.config, a.sequence, a.output, a.gpus, a.resume, a.dry_run, not a.skip_eval)


if __name__ == "__main__":
    main()
