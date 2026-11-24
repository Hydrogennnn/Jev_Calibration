"""Calibration arithmetic, paired forgetting, and sealed offline backfill."""
import csv
import json

import pytest

from evaluation.vqa_cl.calibration import (
    Observation, analyze, calibration_metrics, digest, paired_forgetting,
    read_observations, expected_questions, validate_options,
)
from evaluation.vqa_cl.protocol import BENCHMARK


def observation(index, confidence, correct):
    return Observation(index, "q", str(index), "choice", 10, "a" if correct else "b",
                       "a", confidence, confidence if correct else (1 - confidence) / 9, correct)


def test_ece_arithmetic_direction_and_empty_bins():
    rows = [observation(i, c, ok) for i, (c, ok) in enumerate([
        (.6, True), (.8, False), (.9, True), (1., False)])]
    metrics = calibration_metrics(rows, bins=5)
    assert metrics["ece"] == pytest.approx(.525)
    assert metrics["overconfidence_error"] == pytest.approx(.425)
    assert metrics["underconfidence_error"] == pytest.approx(.1)
    assert metrics["signed_confidence_gap"] == pytest.approx(.325)
    assert metrics["mean_wrong_confidence"] == pytest.approx(.9)
    assert metrics["high_confidence_error_rate"] == pytest.approx(2 / 3)
    assert [row["count"] for row in metrics["reliability_bins"]] == [0, 0, 0, 1, 3]
    assert metrics["reliability_bins"][0]["accuracy"] is None
    boundaries = calibration_metrics([observation(i, c, True) for i, c in enumerate([0., .5, 1.])], bins=2)
    assert [row["count"] for row in boundaries["reliability_bins"]] == [1, 2]
    empty = calibration_metrics([])
    assert empty["count"] == 0 and empty["ece"] is None and empty["mean_wrong_confidence"] is None
    perfect = calibration_metrics([observation(0, 1., True)])
    assert perfect["ece"] == 0 and perfect["mean_wrong_confidence"] is None
    for bins, threshold in [(0, .8), (1.5, .8), (True, .8), (15, float("nan")), (15, 1.1)]:
        with pytest.raises(ValueError):
            validate_options(bins, threshold)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(tmp_path):
    names = ["recognition", "location", "judge"]
    tasks = []
    for name in names:
        data = tmp_path / f"{name}.jsonl"
        records = [{"group_id": str(i), "meta": {"record_id": f"{name}:{i}"},
                    "request": {"state": "test", "questions": {"q": {
                        "type": "choice", "instructions": "Which?", "criteria": {"a": "cat", "b": "dog"}}}},
                    "targets": {"q": {"probabilities": {"a": 1., "b": 0.}}}} for i in range(4)]
        data.write_text("".join(json.dumps(row) + "\n" for row in records))
        tasks.append({"name": name, "test": data.name, "test_sha256": digest(data)})
    sequence = tmp_path / "sequence.json"
    dump(sequence, {"benchmark": BENCHMARK, "tasks": tasks})
    output = tmp_path / "run"
    dump(output / "sequence_run.json", {"sequence_sha256": digest(sequence), "tasks": names})
    cells = {}
    for i, name in enumerate(names):
        stage = output / f"{i:02d}_{name}"
        # Offline analysis uses sealed fingerprints and needs no model checkpoint.
        dump(stage / "stage_complete.json", {"checkpoint_sha256": f"checkpoint-{i}"})
        for j in range(i + 1):
            destination = stage / "eval" / names[j]
            rows = []
            probabilities = [[.7, .8, .3, .2], [.1, .9, .2, .6], [.05, .1, .7, .8]][i]
            for index, p in reversed(list(enumerate(probabilities))):
                rows.append({"record_index": index, "qid": "q", "record_id": f"{names[j]}:{index}",
                    "group_id": str(index), "task": "choice", "target": {"a": 1., "b": 0.},
                    "probabilities": {"a": p, "b": 1 - p}, "answer": {"confidence": 0.123}})
            dump(destination / "metrics.json", {"checkpoint_sha256": f"checkpoint-{i}",
                "data_sha256": tasks[j]["test_sha256"], "temperature": 1.,
                "metrics": {"overall": {"accuracy": .5}}})
            (destination / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            dump(destination / "evaluation_complete.json", {"checkpoint_sha256": f"checkpoint-{i}",
                "data_sha256": tasks[j]["test_sha256"], "metrics_sha256": digest(destination / "metrics.json"),
                "predictions_sha256": digest(destination / "predictions.jsonl")})
            cells[i, j] = destination
    return sequence, output, cells


def test_offline_task_matrix_and_same_sample_forgetting(tmp_path):
    sequence, output, cells = fixture_run(tmp_path)
    originals = {path: path.read_bytes() for folder in cells.values() for path in folder.iterdir()}
    matrix = analyze(sequence, output, bins=5)
    first = matrix[0][0]
    assert first["overall"]["mean_confidence"] == pytest.approx(.75)  # Not answer.confidence!
    assert first["forgetting_from_learning"]["forgotten"]["count"] == 0
    assert not first["forgetting_from_previous_stage"]["available"]
    forgetting = matrix[1][0]["forgetting_from_learning"]
    assert all(forgetting[name]["count"] == 1 for name in (
        "forgotten", "retained_correct", "persistent_wrong", "newly_correct"))
    lost = forgetting["forgotten"]
    assert lost["mean_current_confidence"] == pytest.approx(.9)
    assert lost["mean_confidence_change"] == pytest.approx(.2)
    assert lost["mean_gold_probability_change"] == pytest.approx(-.6)
    assert lost["fraction_of_baseline_correct"] == .5
    assert lost["high_confidence_fraction"] == 1
    assert matrix[2][0]["forgetting_from_learning"]["forgotten"]["count"] == 2
    assert matrix[2][0]["forgetting_from_previous_stage"]["forgotten"]["count"] == 1
    forgotten = [json.loads(line) for line in (cells[2, 0] / "forgotten_samples.jsonl").read_text().splitlines()]
    assert [row["record_index"] for row in forgotten] == [0, 1]
    assert [row["forgotten_from_previous_stage"] for row in forgotten] == [False, True]
    report = json.loads((output / "calibration_matrix.json").read_text())
    assert report["cells"] == matrix and report["completed_cells"] == 6
    assert report["ece_matrix"][0][1] is None
    assert report["macro_task_ece_by_stage"][-1] == pytest.approx(
        sum(cell["overall"]["ece"] for cell in matrix[-1]) / 3)
    with (output / "calibration_tasks.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 6 and csv_rows[3]["forgotten_count"] == "2"
    assert analyze(sequence, output, bins=5) == matrix
    assert all(path.read_bytes() == content for path, content in originals.items())
    # Different bins regenerate sidecars, while original signed results remain intact.
    analyze(sequence, output, bins=1, threshold=.95)
    assert json.loads((output / "calibration_matrix.json").read_text())["bins"] == 1
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_partial_run_does_not_invent_baselines(tmp_path):
    sequence, output, cells = fixture_run(tmp_path)
    (cells[0, 0] / "evaluation_complete.json").unlink()
    (cells[1, 1] / "evaluation_complete.json").unlink()
    matrix = analyze(sequence, output)
    assert matrix[0][0] is None and matrix[1][1] is None
    assert not matrix[2][0]["forgetting_from_learning"]["available"]
    assert matrix[2][0]["forgetting_from_previous_stage"]["available"]
    assert not matrix[2][1]["forgetting_from_previous_stage"]["available"]
    report = json.loads((output / "calibration_matrix.json").read_text())
    assert report["completed_cells"] == 4
    assert report["macro_task_ece_by_stage"][:2] == [None, None]


@pytest.mark.parametrize("change,match", [
    ("prediction", "Unsealed"), ("data", "test data changed"),
    ("plan", "Run plan"), ("temperature", "different temperatures"),
    ("accuracy", "Recomputed accuracy"),
])
def test_reject_changed_sources(tmp_path, change, match):
    sequence, output, cells = fixture_run(tmp_path)
    cell = cells[1, 0]
    if change == "prediction":
        with (cell / "predictions.jsonl").open("a") as stream:
            stream.write("\n")
    elif change == "data":
        with (tmp_path / "recognition.jsonl").open("a") as stream:
            stream.write("\n")
    elif change == "plan":
        dump(output / "sequence_run.json", {"tasks": [], "sequence_sha256": digest(sequence)})
    else:
        report = json.loads((cell / "metrics.json").read_text())
        if change == "temperature":
            report["temperature"] = 2.
        else:
            report["metrics"]["overall"]["accuracy"] = 1.
        dump(cell / "metrics.json", report)
        marker = json.loads((cell / "evaluation_complete.json").read_text())
        marker["metrics_sha256"] = digest(cell / "metrics.json")
        dump(cell / "evaluation_complete.json", marker)
    with pytest.raises(ValueError, match=match):
        analyze(sequence, output)


@pytest.mark.parametrize("change,match", [
    ("missing", "Incomplete"), ("duplicate", "Duplicate"), ("identity", "identity/type/target"),
    ("nan", "finite"), ("negative", "finite"), ("mass", "sum to 1"), ("keys", "candidates"),
])
def test_reject_bad_predictions(tmp_path, change, match):
    _, _, cells = fixture_run(tmp_path)
    path = cells[0, 0] / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "identity":
        rows[0]["record_id"] = "different sample"
    else:
        rows[0]["probabilities"] = {"nan": {"a": float("nan"), "b": .5},
            "negative": {"a": -.1, "b": 1.1}, "mass": {"a": .1, "b": .2},
            "keys": {"a": .5, "c": .5}}[change]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        read_observations(path, expected_questions(tmp_path / "recognition.jsonl"))


def test_pair_coverage_and_zero_baseline_correct():
    old = {(0, "q"): observation(0, .9, False)}
    result = paired_forgetting(old, old, .8)
    assert result["forgotten"]["fraction_of_baseline_correct"] is None
    assert result["forgotten"]["mean_current_confidence"] is None
    with pytest.raises(ValueError, match="coverage"):
        paired_forgetting(old, {}, .8)
