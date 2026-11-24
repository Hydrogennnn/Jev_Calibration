import json
from pathlib import Path

import pytest

from evaluation.vqa_cl.prepare import AnswerPool, answer_key, convert, digest, prepare
from evaluation.vqa_cl.protocol import BENCHMARK, TASKS
from evaluation.vqa_cl.run import run
from valen.data.schema import read_jsonl


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def make_source(root):
    for task in TASKS:
        answers = ["cat", "dog", "."]
        if task == "judge":
            answers = ["yes", "no", "maybe"]
        if task == "count":
            answers = ["0", "2", "many"]
        train = [{"id": 7, "image": "./COCO2014/train2014/a.jpg", "conversations": [
            {"from": "human", "value": f"<image>\nQuestion {i}?\nAnswer the question using a single word or phrase."},
            {"from": "gpt", "value": answer}]} for i, answer in enumerate(answers)]
        # Real author training rows can contain multiple questions for one image.
        train[0]["conversations"] += [{"from": "human", "value": "Another question?"},
                                      {"from": "gpt", "value": answers[1]}]
        test_answers = {"judge": ["yes", "other"], "count": ["2", "21"]}.get(task, ["TEST-ONLY", "."])
        test = [{"question_id": i, "image": "./COCO2014/val2014/b.jpg",
                 "text": "Question?", "answer": answer} for i, answer in enumerate(test_answers)]
        write(root / "train" / f"train_q_{task}.json", train)
        write(root / "test" / f"test_q_{task}.json", test)


def test_ten_tasks_native_splits_primitives_and_no_test_pool_leakage(tmp_path):
    source, images = tmp_path / "CL4VQA", tmp_path / "COCO"
    make_source(source)
    for split, name in (("train2014", "a.jpg"), ("val2014", "b.jpg")):
        (images / split).mkdir(parents=True)
        (images / split / name).write_bytes(b"image existence fixture")
    a, b = tmp_path / "a", tmp_path / "b"
    first = prepare(source, images, a)
    assert prepare(source, images, b) == first
    assert first["benchmark"] == BENCHMARK
    assert [task["name"] for task in first["tasks"]] == list(TASKS)
    assert first["cross_task_train_eval_overlapping_groups"] == 0
    for task in first["tasks"]:
        assert task["counts"] == {"train": 4, "test": 2}
        assert task["source_rows"] == {"train": 3, "test": 2}
        folder = a / task["name"]
        for split in ("train", "test"):
            assert (folder / f"{split}.jsonl").read_bytes() == (b / task["name"] / f"{split}.jsonl").read_bytes()
            rows = read_jsonl(folder / f"{split}.jsonl")
            assert len({r["meta"]["record_id"] for r in rows}) == len(rows)
            for row in rows:
                meta = row["meta"]
                probabilities = row["targets"]["q0"]["probabilities"]
                winner = max(probabilities, key=probabilities.get)
                assert answer_key(meta["answer_map"][winner]) == answer_key(meta["gold_answer"])
                prompt = json.dumps(row["request"])
                assert "gold_answer" not in prompt and "record_id" not in prompt
        pool = json.loads((folder / "answer_pool.json").read_text())
        assert "TEST-ONLY" not in pool
    assert first["tasks"][2]["primitive_counts"]["test"] == {"noul": 1, "choice": 1}
    assert first["tasks"][4]["primitive_counts"]["test"] == {"score": 1, "choice": 1}
    assert read_jsonl(a / "recognition/train.jsonl")[0]["request"]["state"]["messages"][0]["content"][1]["image_url"]["url"] == str(images / "train2014/a.jpg")


def test_smoke_reverse_choice_ablation_and_atomic_failures(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    manifest = prepare(source, tmp_path, tmp_path / "smoke", tasks=["Cou.", "Jud."],
                       limit=2, primitives="choice", allow_missing_images=True)
    assert manifest["smoke_limit"] == 2
    assert [t["name"] for t in manifest["tasks"]] == ["count", "judge"]
    assert all(t["primitive_counts"]["train"] == {"choice": 2} for t in manifest["tasks"])
    with pytest.raises(FileNotFoundError):
        prepare(source, tmp_path, tmp_path / "missing-images", tasks=["recognition"])
    assert not (tmp_path / "missing-images").exists()
    rows = json.loads((source / "train/train_q_recognition.json").read_text())
    rows[0]["conversations"][1]["value"] = " "
    write(source / "train/train_q_recognition.json", rows)
    with pytest.raises(ValueError, match="row 0"):
        prepare(source, tmp_path, tmp_path / "invalid", tasks=["recognition"], allow_missing_images=True)
    assert not (tmp_path / "invalid").exists()


def test_case_and_punctuation_are_displayed_but_equivalent_keys_not_distractors():
    import random
    pool = AnswerPool([{"answer": value} for value in ["Red", "red", ".", "Blue."]])
    values = pool.choices("RED", 8, random.Random(42))
    assert values == ["RED", ".", "Blue."] or values == ["RED", "Blue.", "."]
    assert len(pool.keys) == 3 and "." in pool.keys


def test_sequence_initializes_consolidated_checkpoints_and_resumes(tmp_path, monkeypatch):
    source, prepared, output = tmp_path / "source", tmp_path / "data", tmp_path / "output"
    make_source(source)
    prepare(source, tmp_path, prepared, allow_missing_images=True)
    config = Path(__file__).resolve().parents[2] / "configs/train/qwen/vqa_cl_moe.json"
    calls = []

    def execute(cmd):
        calls.append(cmd)
        if "valen.train" in cmd:
            stage = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text())
            index = stage["cl_moe_task"]
            assert stage["cl_moe"]["max_tasks"] == 10
            assert stage["global_batch_size"] == 16 and stage["lr_scheduler"] == "cosine"
            if index:
                assert cmd[cmd.index("--initialize") + 1] == str(output / f"{index - 1:02d}_{TASKS[index - 1]}" / "consolidated")
            checkpoint = Path(stage["output"]) / "consolidated/checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"stage {index}".encode())
        else:
            destination = Path(cmd[cmd.index("--output") + 1])
            checkpoint = Path(cmd[cmd.index("--checkpoint") + 1])
            data = Path(cmd[cmd.index("--data") + 1])
            write(destination / "metrics.json", {"checkpoint_sha256": digest(checkpoint / "checkpoint.pt"),
                "data_sha256": digest(data), "temperature": 1., "metrics": {"overall": {"accuracy": .5}}})
            predictions = []
            for index, record in enumerate(read_jsonl(data)):
                target = record["targets"]["q0"]["probabilities"]
                gold = max(target, key=target.get)
                predicted = gold if index == 0 else next(key for key in target if key != gold)
                predictions.append({"record_index": index, "qid": "q0", "record_id": record["meta"]["record_id"],
                    "group_id": record["group_id"], "task": record["request"]["questions"]["q0"]["type"],
                    "target": target, "probabilities": {key: .9 if key == predicted else .1 / (len(target) - 1) for key in target}})
            (destination / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))

    monkeypatch.setattr("valen.continual.run_command", execute)
    result = run(config, prepared / "sequence.json", output, gpus=4)
    assert len(calls) == 10 + 55 and len(result) == 10
    metrics = json.loads((output / "continual_metrics.json").read_text())
    assert metrics["average_performance"] == .5 and metrics["average_forgetting"] == 0
    calibration = json.loads((output / "calibration_matrix.json").read_text())
    assert calibration["completed_cells"] == calibration["expected_cells"] == 55
    assert calibration["ece_matrix"][-1] == pytest.approx([.4] * 10)
    calls.clear()
    assert run(config, prepared / "sequence.json", output, gpus=4, resume=True) == result
    assert not calls
    assert json.loads((output / "calibration_matrix.json").read_text()) == calibration
    with pytest.raises(ValueError, match="divisible"):
        run(config, prepared / "sequence.json", tmp_path / "bad", gpus=3, dry_run=True)
    with (prepared / "recognition/test.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="data changed"):
        run(config, prepared / "sequence.json", output, gpus=4, resume=True)
