"""Task-level top-label calibration and paired forgetting analysis.

Only sealed evaluation predictions are used. No temperature fitting, model load,
or changes to the original metrics/predictions/completion markers are performed.
"""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from valen.data.schema import candidates, target_distribution
from .protocol import BENCHMARK, task_order


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_options(bins, threshold):
    if type(bins) is not int or not 1 <= bins <= 1000:
        raise ValueError("ECE bins must be an integer in [1, 1000]")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("High-confidence threshold must be in [0, 1]")


@dataclass(frozen=True)
class Observation:
    record_index: int
    qid: str
    record_id: str | None
    kind: str
    candidate_count: int
    predicted: str
    gold: str
    confidence: float
    gold_probability: float
    correct: bool


def average(values):
    return math.fsum(values) / len(values) if values else None


def calibration_metrics(observations, bins=15, threshold=.8):
    """Equal-width ECE, with [lo, hi) bins and a closed upper endpoint at 1."""
    validate_options(bins, threshold)
    observations = list(observations)
    n = len(observations)
    buckets = [[] for _ in range(bins)]
    for item in observations:
        buckets[min(int(item.confidence * bins), bins - 1)].append(item)
    reliability = []
    for index, bucket in enumerate(buckets):
        confidence = average([item.confidence for item in bucket])
        accuracy = average([float(item.correct) for item in bucket])
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
            "count": len(bucket), "mean_confidence": confidence, "accuracy": accuracy,
            "signed_gap": confidence - accuracy if bucket else None})
    wrong = [item for item in observations if not item.correct]
    high = [item for item in observations if item.confidence >= threshold]
    high_wrong = sum(not item.correct for item in high)
    over = math.fsum(row["count"] * max(row["signed_gap"], 0) for row in reliability if row["count"])
    under = math.fsum(row["count"] * max(-row["signed_gap"], 0) for row in reliability if row["count"])
    confidence = average([item.confidence for item in observations])
    accuracy = average([float(item.correct) for item in observations])
    return {"count": n, "accuracy": accuracy, "mean_confidence": confidence,
        "ece": (over + under) / n if n else None,
        "overconfidence_error": over / n if n else None,
        "underconfidence_error": under / n if n else None,
        "signed_confidence_gap": confidence - accuracy if n else None,
        "wrong_count": len(wrong), "mean_wrong_confidence": average([item.confidence for item in wrong]),
        "mean_correct_confidence": average([item.confidence for item in observations if item.correct]),
        "high_confidence_count": len(high), "high_confidence_wrong_count": high_wrong,
        "high_confidence_error_rate": high_wrong / len(high) if high else None,
        "high_confidence_wrong_fraction": high_wrong / n if n else None,
        "reliability_bins": reliability}


def paired_forgetting(baseline, current, threshold):
    """Compare the SAME questions, keeping losses separate from later learning."""
    if baseline is None:
        return {"available": False, "reason": "Baseline evaluation is unavailable"}
    if set(baseline) != set(current):
        raise ValueError("Cannot compare stages with different question coverage")
    cohorts = {"forgotten": [], "retained_correct": [], "persistent_wrong": [], "newly_correct": []}
    for key, now in current.items():
        old = baseline[key]
        name = ("retained_correct" if now.correct else "forgotten") if old.correct else (
            "newly_correct" if now.correct else "persistent_wrong")
        cohorts[name].append((old, now))
    result = {"available": True, "count": len(current), "baseline_correct_count": sum(o.correct for o in baseline.values())}
    for name, pairs in cohorts.items():
        n = len(pairs)
        high = sum(now.confidence >= threshold for _, now in pairs)
        result[name] = {"count": n, "fraction_of_all": n / len(current) if current else None,
            "mean_baseline_confidence": average([old.confidence for old, _ in pairs]),
            "mean_current_confidence": average([now.confidence for _, now in pairs]),
            "mean_confidence_change": average([now.confidence - old.confidence for old, now in pairs]),
            "mean_baseline_gold_probability": average([old.gold_probability for old, _ in pairs]),
            "mean_current_gold_probability": average([now.gold_probability for _, now in pairs]),
            "mean_gold_probability_change": average([now.gold_probability - old.gold_probability for old, now in pairs]),
            "confidence_increased_count": sum(now.confidence > old.confidence for old, now in pairs),
            "high_confidence_count": high, "high_confidence_fraction": high / n if n else None}
    correct = result["baseline_correct_count"]
    result["forgotten"]["fraction_of_baseline_correct"] = len(cohorts["forgotten"]) / correct if correct else None
    return result


def expected_questions(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        index = 0
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            for qid, question in record["request"]["questions"].items():
                keys = [key for key, _ in candidates(question)]
                target = target_distribution(record.get("targets", {}).get(qid), keys)
                if target is None:
                    continue
                if target.count(1.) != 1 or any(value not in (0., 1.) for value in target):
                    raise ValueError("Top-label calibration requires hard one-hot labels")
                result[index, qid] = {"kind": question["type"], "target": dict(zip(keys, target)),
                    "record_id": record.get("meta", {}).get("record_id"), "group_id": record["group_id"]}
            index += 1
    if not result:
        raise ValueError(f"No labeled questions: {path}")
    return result


def read_observations(path, expected):
    observations = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                index, qid = row["record_index"], row["qid"]
                if type(index) is not int or not isinstance(qid, str):
                    raise ValueError("Invalid record_index/qid")
                key = index, qid
                if key in observations or key not in expected:
                    raise ValueError("Duplicate or unexpected question")
                reference = expected[key]
                if (row["target"] != reference["target"] or row["task"] != reference["kind"]
                        or row.get("record_id") != reference["record_id"] or row["group_id"] != reference["group_id"]):
                    raise ValueError("Prediction identity/type/target differs from prepared data")
                probs = row["probabilities"]
                if not isinstance(probs, dict) or set(probs) != set(reference["target"]):
                    raise ValueError("Prediction candidates differ from prepared data")
                if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
                       or not 0 <= p <= 1 for p in probs.values()):
                    raise ValueError("Probabilities must be finite numbers in [0, 1]")
                mass = math.fsum(probs.values())
                if not math.isclose(mass, 1., rel_tol=0., abs_tol=1e-5):
                    raise ValueError("Probabilities must sum to 1")
                # Only correct float32 summation roundoff. Preserve saved key order
                # for argmax ties, matching the original evaluation convention.
                probs = {key: value / mass for key, value in probs.items()}
                predicted = max(probs, key=probs.get)
                gold = next(key for key, value in reference["target"].items() if value == 1)
                observations[index, qid] = Observation(index, qid, row.get("record_id"), row["task"],
                    len(probs), predicted, gold, probs[predicted], probs[gold], predicted == gold)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if set(observations) != set(expected):
        raise ValueError(f"Incomplete prediction coverage: {path}; expected {len(expected)}, got {len(observations)}")
    return observations


class CalibrationTracker:
    """One tracker per sequence pass, used both online and for offline backfill."""
    def __init__(self, sequence, output, bins=15, threshold=.8, *, generic=False):
        validate_options(bins, threshold)
        self.sequence_path, self.output = Path(sequence).resolve(), Path(output).resolve()
        self.sequence = json.loads(self.sequence_path.read_text())
        self.generic = generic
        if not generic and self.sequence.get("benchmark") != BENCHMARK:
            raise ValueError("Expected a CL4VQA sequence")
        self.tasks = [task["name"] for task in self.sequence["tasks"]]
        if generic:
            if (self.sequence.get("format") != "coin_decision_v1" or not self.tasks
                    or any(not isinstance(name, str) or not name or name in {".", ".."}
                           or Path(name).name != name for name in self.tasks)
                    or len(set(self.tasks)) != len(self.tasks)):
                raise ValueError("Expected a native continual sequence with unique task names")
        elif task_order(self.tasks) != self.tasks:
            raise ValueError("Expected canonical CL4VQA task names")
        self.bins, self.threshold = bins, threshold
        self.sequence_hash = digest(self.sequence_path)
        self.matrix = [[None] * len(self.tasks) for _ in self.tasks]
        self.expected, self.diagonal, self.previous = {}, {}, {}
        self.temperature = None

    def destination(self, i, j):
        return self.output / f"{i:02d}_{self.tasks[i]}" / "eval" / self.tasks[j]

    def observe(self, i, j, destination, report=None):
        destination = Path(destination).resolve()
        if not 0 <= j <= i < len(self.tasks) or destination != self.destination(i, j):
            raise ValueError("Unexpected stage/task evaluation path")
        plan = json.loads((self.output / "sequence_run.json").read_text())
        if plan["sequence_sha256"] != self.sequence_hash or plan["tasks"] != self.tasks:
            raise ValueError("Run plan does not match the requested sequence")
        stage = destination.parent.parent
        stage_marker = json.loads((stage / "stage_complete.json").read_text())
        marker = json.loads((destination / "evaluation_complete.json").read_text())
        expected_marker = {"checkpoint_sha256": stage_marker["checkpoint_sha256"],
            "data_sha256": self.sequence["tasks"][j]["test_sha256"],
            "metrics_sha256": digest(destination / "metrics.json"),
            "predictions_sha256": digest(destination / "predictions.jsonl")}
        if marker != expected_marker:
            raise ValueError(f"Unsealed or changed evaluation: {destination}")
        report = json.loads((destination / "metrics.json").read_text())
        if any(report[key] != marker[key] for key in ("checkpoint_sha256", "data_sha256")):
            raise ValueError("Evaluation report checkpoint/data mismatch")
        temperature = report.get("temperature")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Evaluation must record a positive temperature")
        if self.temperature is not None and self.temperature != temperature:
            raise ValueError("Cannot compare evaluations using different temperatures")
        self.temperature = temperature
        if j not in self.expected:
            data = (self.sequence_path.parent / self.sequence["tasks"][j]["test"]).resolve()
            if digest(data) != marker["data_sha256"]:
                raise ValueError(f"Prepared test data changed: {data}")
            self.expected[j] = expected_questions(data)
        current = read_observations(destination / "predictions.jsonl", self.expected[j])
        metrics = calibration_metrics(current.values(), self.bins, self.threshold)
        if not math.isclose(metrics["accuracy"], report["metrics"]["overall"]["accuracy"], abs_tol=1e-6):
            raise ValueError("Recomputed accuracy differs from sealed evaluation")
        baseline = current if i == j else self.diagonal.get(j)
        prior = self.previous.get(j)
        previous = prior[1] if prior and prior[0] == i - 1 else None
        cell = {"stage_index": i, "stage_task": self.tasks[i], "task_index": j, "task": self.tasks[j],
            "options": {"bins": self.bins, "high_confidence_threshold": self.threshold},
            "sources": marker, "temperature": temperature, "overall": metrics,
            "by_primitive": {kind: calibration_metrics([o for o in current.values() if o.kind == kind], self.bins, self.threshold)
                             for kind in sorted({o.kind for o in current.values()})},
            "by_candidate_count": {str(count): calibration_metrics([o for o in current.values() if o.candidate_count == count], self.bins, self.threshold)
                                   for count in sorted({o.candidate_count for o in current.values()})},
            "forgetting_from_learning": paired_forgetting(baseline, current, self.threshold),
            "forgetting_from_previous_stage": paired_forgetting(previous, current, self.threshold)}
        self.write_forgotten(destination, current, baseline, previous)
        write_json(destination / "calibration.json", cell)
        self.matrix[i][j] = cell
        if i == j:
            self.diagonal[j] = current
        self.previous[j] = i, current
        self.write_summary()
        print(json.dumps({"event": "task_calibration", "stage": i, "task": self.tasks[j],
                          "ece": metrics["ece"], "accuracy": metrics["accuracy"],
                          "mean_wrong_confidence": metrics["mean_wrong_confidence"]}), flush=True)

    def write_forgotten(self, destination, current, baseline, previous):
        path = destination / "forgotten_samples.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for key, now in sorted(current.items()):
                if now.correct:
                    continue
                old, prior = (baseline or {}).get(key), (previous or {}).get(key)
                if not ((old and old.correct) or (prior and prior.correct)):
                    continue
                row = {"record_index": now.record_index, "qid": now.qid, "record_id": now.record_id,
                    "primitive": now.kind, "candidate_count": now.candidate_count, "gold": now.gold,
                    "predicted": now.predicted, "confidence": now.confidence,
                    "gold_probability": now.gold_probability, "high_confidence": now.confidence >= self.threshold}
                for name, before in (("learning", old), ("previous_stage", prior)):
                    row[f"forgotten_from_{name}"] = bool(before and before.correct)
                    row[f"{name}_confidence"] = before.confidence if before else None
                    row[f"{name}_gold_probability"] = before.gold_probability if before else None
                    row[f"confidence_change_from_{name}"] = now.confidence - before.confidence if before else None
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)

    def write_summary(self):
        def matrix(metric):
            return [[cell["overall"][metric] if cell else None for cell in row] for row in self.matrix]
        summary = {"format": "continual_calibration_v1" if self.generic else "vqa_cl_calibration_v1", "tasks": self.tasks,
            "sequence_sha256": self.sequence_hash, "bins": self.bins,
            "binning": "equal_width_[lower,upper)_last_includes_1", "confidence": "max_saved_probability",
            "correctness": "argmax_candidate_equals_hard_target_in_saved_key_order",
            "high_confidence_threshold": self.threshold, "temperature": self.temperature,
            "smoke_limit": self.sequence.get("smoke_limit"),
            "completed_cells": sum(cell is not None for row in self.matrix for cell in row),
            "expected_cells": len(self.tasks) * (len(self.tasks) + 1) // 2,
            "cells": self.matrix}
        for metric in ("ece", "accuracy", "mean_confidence", "signed_confidence_gap",
                       "overconfidence_error", "mean_wrong_confidence", "high_confidence_error_rate"):
            summary[f"{metric}_matrix"] = matrix(metric)
        summary["macro_task_ece_by_stage"] = [average([cell["overall"]["ece"] for cell in row[:i + 1]])
            if all(cell is not None for cell in row[:i + 1]) else None for i, row in enumerate(self.matrix)]
        write_json(self.output / "calibration_matrix.json", summary)
        path = self.output / "calibration_tasks.csv"
        temporary = path.with_suffix(".csv.tmp")
        fields = ["stage_index", "stage_task", "task", "count", "accuracy", "ece", "mean_confidence",
                  "overconfidence_error", "underconfidence_error", "signed_confidence_gap", "wrong_count",
                  "mean_wrong_confidence", "high_confidence_error_rate", "forgotten_count",
                  "forgotten_fraction_of_baseline_correct", "forgotten_mean_confidence",
                  "forgotten_mean_confidence_change", "forgotten_mean_gold_probability_change",
                  "forgotten_high_confidence_fraction", "newly_forgotten_count"]
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.matrix:
                for cell in row:
                    if cell is None:
                        continue
                    item = {key: cell[key] for key in ("stage_index", "stage_task", "task")}
                    item.update({key: cell["overall"][key] for key in fields if key in cell["overall"]})
                    forgotten = cell["forgetting_from_learning"].get("forgotten", {})
                    for column, key in (("forgotten_count", "count"), ("forgotten_fraction_of_baseline_correct", "fraction_of_baseline_correct"),
                        ("forgotten_mean_confidence", "mean_current_confidence"), ("forgotten_mean_confidence_change", "mean_confidence_change"),
                        ("forgotten_mean_gold_probability_change", "mean_gold_probability_change"), ("forgotten_high_confidence_fraction", "high_confidence_fraction")):
                        item[column] = forgotten.get(key)
                    item["newly_forgotten_count"] = cell["forgetting_from_previous_stage"].get("forgotten", {}).get("count")
                    writer.writerow(item)
        temporary.replace(path)


def analyze(sequence, output, bins=15, threshold=.8):
    tracker = CalibrationTracker(sequence, output, bins, threshold)
    for i in range(len(tracker.tasks)):
        for j in range(i + 1):
            destination = tracker.destination(i, j)
            if (destination / "evaluation_complete.json").exists():
                tracker.observe(i, j, destination)
    if not any(cell for row in tracker.matrix for cell in row):
        raise ValueError("No completed, sealed task evaluations found")
    return tracker.matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="data/vqa-cl-decision/sequence.json")
    parser.add_argument("--run", required=True, help="Existing continual output directory")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--high-confidence-threshold", type=float, default=.8)
    args = parser.parse_args()
    analyze(args.sequence, args.run, args.bins, args.high_confidence_threshold)


if __name__ == "__main__":
    main()
