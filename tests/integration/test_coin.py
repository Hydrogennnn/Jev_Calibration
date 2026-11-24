import json
from pathlib import Path

import pytest

from evaluation.coin.prepare import (AnswerPool, TASKS, answer_counts, convert, prepare, read_samples,
                                     science_options, box_iou)
from valen.continual import completed_evaluation, evaluation_signature, run_sequence, summarize_matrix
from valen.data.schema import read_jsonl


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_official_multiturn_keeps_all_questions_and_no_teacher_history(tmp_path):
    source = tmp_path / "train.json"
    write(source, [{"id": "image-1", "image": "a.jpg", "conversations": [
        {"from": "human", "value": "<image>\nIs it cloudy?\nAnswer the question using a single word or phrase."},
        {"from": "gpt", "value": "Yes"}, {"from": "human", "value": "What color?"},
        {"from": "gpt", "value": "Red"}]}])
    samples = read_samples(source, "GQA", tmp_path)
    assert len(samples) == 2 and samples[0]["group"] == samples[1]["group"]
    assert samples[1]["text"] == "What color?"
    pool = AnswerPool(samples)
    record = convert(samples[0], "GQA", pool, 4, 42, "train")
    assert set(record["request"]["questions"]["q0"]["criteria"].values()) == {"yes", "no"}
    assert "targets" not in record["request"]
    assert "Is it cloudy?" == record["request"]["questions"]["q0"]["instructions"]


def make_source(root):
    for task in TASKS:
        train, test = [], []
        for i in range(15):
            question = "What is shown?"
            answer = ["cat", "dog", "bird", "cow"][i % 4]
            if task == "ScienceQA":
                question += "\nA. cat\nB. dog\nAnswer with the option's letter from the given choices directly."
                answer = "A"
            if task == "Grounding":
                question = "Please provide the bounding box coordinate of the region this sentence describes: cat."
                answer = "[0.1, 0.1, 0.6, 0.7]"
            train.append(dict(id=str(i), image=f"image-{i}.jpg", conversations=[
                dict(**{"from": "human"}, value="<image>\n" + question),
                dict(**{"from": "gpt"}, value=answer)]))
            item = dict(question_id=str(i), image=f"test-{i}.jpg", text=question)
            item["answer_bbox" if task == "Grounding" else "answer"] = answer
            test.append(item)
        write(root / task / "train.json", train)
        write(root / task / ("val.json" if task in {"VQAv2", "VizWiz", "TextVQA"} else "test.json"), test)


def test_all_eight_tasks_reproducible_conversion_and_sequence_dry_run(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    a, b = tmp_path / "a", tmp_path / "b"
    first = prepare(source, tmp_path, a, choices=4, allow_missing_images=True)
    second = prepare(source, tmp_path, b, choices=4, allow_missing_images=True)
    assert first == second
    for task in TASKS:
        assert (a / task / "train.jsonl").read_bytes() == (b / task / "train.jsonl").read_bytes()
        records = read_jsonl(a / task / "test.jsonl")
        assert len(records) == 15
        if task == "Grounding":
            for record in records:
                gold = record["meta"]["gold_bbox"]
                boxes = record["meta"]["candidate_boxes"]
                label = max(record["targets"]["q0"]["probabilities"], key=record["targets"]["q0"]["probabilities"].get)
                assert box_iou(gold, boxes[label]) == 1
                assert all(box_iou(gold, box) < .3 for key, box in boxes.items() if key != label)
    config = tmp_path / "config.json"
    write(config, dict(model_path="unused", finetuning_type="cl_moe", cl_moe=dict(max_tasks=8), data="unused"))
    output = tmp_path / "run"
    run_sequence(config, a / "sequence.json", output, dry_run=True)
    last = json.loads((output / "07_OCRVQA" / "stage_config.json").read_text())
    assert last["cl_moe_task"] == 7
    run_sequence(config, a / "sequence.json", output, dry_run=True, resume=True)
    with pytest.raises(ValueError, match="Output exists"):
        run_sequence(config, a / "sequence.json", output, dry_run=True)
    (a / "GQA" / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="data changed"):
        run_sequence(config, a / "sequence.json", output, dry_run=True, resume=True)


def test_missing_labels_fail_atomically_and_test_answer_not_in_pool(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    test = source / "GQA" / "test.json"
    rows = json.loads(test.read_text())
    rows[0].pop("answer")
    write(test, rows)
    out = tmp_path / "converted"
    with pytest.raises(ValueError, match="missing labels"):
        prepare(source, tmp_path, out, tasks=["GQA"], allow_missing_images=True)
    assert not out.exists()
    rows[0]["answer"] = "test-only-answer"
    write(test, rows)
    prepare(source, tmp_path, out, tasks=["GQA"], allow_missing_images=True)
    assert "test-only-answer" not in json.loads((out / "GQA" / "answer_pool.json").read_text())
    # Gold included only in its own predefined choice question.
    assert "test-only-answer" in read_jsonl(out / "GQA" / "test.jsonl")[0]["request"]["questions"]["q0"]["criteria"].values()


def test_image_group_holdout_is_consistent_across_tasks(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    out = tmp_path / "heldout"
    manifest = prepare(source, tmp_path, out, tasks=["GQA", "VQAv2"], eval_source="holdout",
                       holdout=.4, allow_missing_images=True)
    assert manifest["cross_task_train_eval_overlapping_groups"] == 0
    train = {r["group_id"] for t in ("GQA", "VQAv2") for r in read_jsonl(out / t / "train.jsonl")}
    test = {r["group_id"] for t in ("GQA", "VQAv2") for r in read_jsonl(out / t / "test.jsonl")}
    assert train.isdisjoint(test)


def test_science_text_only_and_normalized_multivote_targets(tmp_path):
    q, choices, index = science_options("Question\nA. first\nB. second", "B")
    assert (q, choices, index) == ("Question", ["first", "second"], 1)
    sample = dict(id="1", group="1", source_id="1", turn=0, image=None, text="Question\nA. first\nB. second", answer="B")
    record = convert(sample, "ScienceQA", None, 4, 42, "train")
    assert record["meta"]["modality"] == "text"
    sample.update(text="What?", answer=["cat", "cat", "dog"])
    pool = AnswerPool([sample, dict(sample, answer="bird")])
    record = convert(sample, "VQAv2", pool, 4, 42, "test")
    criteria = record["request"]["questions"]["q0"]["criteria"]
    probs = record["targets"]["q0"]["probabilities"]
    assert {criteria[k]: v for k, v in probs.items()} == {"cat": 2/3, "dog": 1/3, "bird": 0}


def test_repeated_source_id_and_root_independent_group_identity(tmp_path):
    path = tmp_path / "rows.json"
    write(path, [{"question_id": "image-id", "image": "./same.jpg", "text": q, "answer": a}
                 for q, a in [("What color?", "red"), ("How many?", "2")]])
    first = read_samples(path, "TextVQA", tmp_path / "host1")
    second = read_samples(path, "TextVQA", tmp_path / "host2")
    assert len(first) == 2 and first[0]["id"] != first[1]["id"]
    assert first[0]["group"] == first[1]["group"] == second[0]["group"]


def test_continual_metrics_distinguish_diagonal_and_maximum_forgetting():
    matrix = [[{"accuracy": .8}, None, None], [{"accuracy": .9}, {"accuracy": .7}, None],
              [{"accuracy": .6}, {"accuracy": .8}, {"accuracy": .9}]]
    report = summarize_matrix(matrix)
    assert report["average_performance"] == pytest.approx(2.3/3)
    assert report["backward_transfer"] == pytest.approx(-.05)
    assert report["average_forgetting"] == pytest.approx(.05)
    assert report["max_past_forgetting"] == pytest.approx(.1)


def test_multivote_candidates_respect_full_capacity_and_reject_empty_answers():
    import random
    pool = AnswerPool([dict(answer=a) for a in ["cat", "dog", "bird", "cow", "fish"]])
    assert pool.choices({"cat": 2, "dog": 1}, 2, random.Random(42)) == ["cat", "dog"]
    with pytest.raises(ValueError, match="Empty answer list"):
        answer_counts([])


def test_textvqa_period_answer_survives_pool_and_target_conversion():
    sample = dict(id="TextVQA:dc8725bb2f38c956:9544:0", group="1",
                  source_id="dc8725bb2f38c956", turn=0, image=None,
                  text="What punctuation is shown?", answer=".")
    assert answer_counts([".", " . ", "...", "Red."]) == {".": 2, "...": 1, "red": 1}
    pool = AnswerPool([sample, dict(sample, answer="comma")])
    record = convert(sample, "TextVQA", pool, 2, 42, "train")
    criteria = record["request"]["questions"]["q0"]["criteria"]
    probs = record["targets"]["q0"]["probabilities"]
    assert {criteria[k]: v for k, v in probs.items()} == {".": 1., "comma": 0.}
    for empty in ["", " \t\n ", ["yes", ""]]:
        with pytest.raises(ValueError, match="Empty answer"):
            answer_counts(empty)


def test_evaluation_cache_requires_success_marker_and_intact_outputs(tmp_path):
    report = dict(checkpoint_sha256="checkpoint", data_sha256="data", metrics={"overall": {"accuracy": .5}})
    write(tmp_path / "metrics.json", report)
    (tmp_path / "predictions.jsonl").write_text('{"prediction": 1}\n')
    assert completed_evaluation(tmp_path, "checkpoint", "data") is None
    signature = evaluation_signature(tmp_path, "checkpoint", "data")
    write(tmp_path / "evaluation_complete.json", signature)
    assert completed_evaluation(tmp_path, "checkpoint", "data") == report
    assert completed_evaluation(tmp_path, "new-checkpoint", "data") is None
    (tmp_path / "metrics.json").write_text('{"partial":')
    assert completed_evaluation(tmp_path, "checkpoint", "data") is None


def test_sequence_rejects_duplicate_task_names_before_creating_output(tmp_path):
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    make_source(source)
    sequence = prepare(source, tmp_path, prepared, tasks=["GQA"], allow_missing_images=True)
    sequence["tasks"] *= 2
    write(prepared / "sequence.json", sequence)
    config, output = tmp_path / "config.json", tmp_path / "run"
    write(config, {"model_path": "unused"})
    with pytest.raises(ValueError, match="unique"):
        run_sequence(config, prepared / "sequence.json", output, dry_run=True)
    assert not output.exists()


@pytest.mark.parametrize("interrupted_stage", ["train", "eval"])
def test_interrupted_command_does_not_mark_stage_complete(tmp_path, monkeypatch, interrupted_stage):
    source, prepared, output = tmp_path / "source", tmp_path / "prepared", tmp_path / "run"
    make_source(source)
    prepare(source, tmp_path, prepared, tasks=["GQA"], allow_missing_images=True)
    config = tmp_path / "config.json"
    write(config, {"model_path": "unused"})

    def run(cmd):
        if interrupted_stage == "train" or "valen.evaluate" in cmd:
            raise SystemExit(130)
        checkpoint = output / "00_GQA" / "latest" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"completed training fixture")

    monkeypatch.setattr("valen.continual.run_command", run)
    with pytest.raises(SystemExit) as error:
        run_sequence(config, prepared / "sequence.json", output)
    assert error.value.code == 130
    assert (output / "00_GQA" / "stage_complete.json").exists() == (interrupted_stage == "eval")
    assert not list(output.rglob("evaluation_complete.json"))
    assert not (output / "continual_metrics.json").exists()
