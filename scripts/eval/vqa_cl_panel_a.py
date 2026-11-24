"""Export panel A from the SAME predictions and metrics used by panel B."""
import csv
import json
import math
from collections import defaultdict

from evaluation.vqa_cl.calibration import Observation, calibration_metrics


def export_panel_a(output, result):
    observations = defaultdict(list)
    # Sidecars export the canonical dataset; they never create a new scenario.
    with (output / "predictions.jsonl").open() as source, (output / "panel_a_predictions.jsonl").open("w") as destination:
        for line in source:
            row = json.loads(line)
            if row["eval_task"] != "recognition":
                continue
            destination.write(line)
            probabilities = row["probabilities"]
            predicted = max(probabilities, key=probabilities.get)
            gold = max(row["target"], key=row["target"].get)
            observations[row["stage_index"]].append(Observation(row["record_index"], row["qid"],
                row["record_id"], row["task"], len(probabilities), predicted, gold,
                probabilities[predicted], probabilities[gold], predicted == gold))
    task_index = result["tasks"].index("recognition")
    stages = []
    for stage, matrix_row in enumerate(result["cells"]):
        cell = matrix_row[task_index]
        if cell is None:
            continue
        metrics = calibration_metrics(observations[stage], result["ece_bins"], result["high_confidence_threshold"])
        for name in ("accuracy", "mean_confidence", "ece", "underconfidence_error"):
            if not math.isclose(metrics[name], cell["overall"][name], abs_tol=1e-12):
                raise ValueError(f"Stage {stage}: predictions and calibration matrix disagree on {name}")
        gap = metrics["mean_confidence"] - metrics["accuracy"]
        if not math.isclose(metrics["ece"], gap + 2 * metrics["underconfidence_error"], abs_tol=1e-12):
            raise ValueError("ECE/gap consistency check failed")
        stages.append({"stage_index": stage, "stage_task": cell["stage_task"], "overall": metrics})
    panel = {"synthetic": True, "source": "calibration_matrix.json and predictions.jsonl",
        "independent_of_panel_b": False, "task": "recognition", "stages": stages,
        "relationship": "ECE = mean_confidence - accuracy + 2 * underconfidence_error"}
    (output / "panel_a.json").write_text(json.dumps(panel, indent=2) + "\n")
    with (output / "panel_a.csv").open("w", newline="") as stream:
        fields = ["synthetic", "stage_index", "stage_task", "accuracy", "mean_confidence", "ece",
                  "signed_confidence_gap", "underconfidence_error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            writer.writerow({"synthetic": True, "stage_index": stage["stage_index"], "stage_task": stage["stage_task"],
                **{key: stage["overall"][key] for key in fields[3:]}})
    return panel
