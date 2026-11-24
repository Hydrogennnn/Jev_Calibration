"""Small real Qwen runs for the new CL4VQA training profile (no GPU required)."""
import json
from pathlib import Path

import pytest
import torch

from test_published_finetuning import published, options
from evaluation.vqa_cl.protocol import BENCHMARK
from evaluation.vqa_cl.run import run as run_sequence
from valen.training.checkpoint import file_sha256
from valen.training.runner import run


def records(kind):
    question = {"type": kind, "instructions": "Which animal"}
    if kind == "choice":
        question["criteria"] = {"a": "cat", "b": "dog"}
        target = {"a": 1., "b": 0.}
    elif kind == "noul":
        target = {"true": 1., "false": 0.}
    else:
        question["criteria"] = ["0", "1", "2"]
        target = {"0": 0., "1": 1., "2": 0.}
    return [{"group_id": str(i), "request": {"state": "cat " * (i + 1), "questions": {"q": question}},
             "targets": {"q": {"probabilities": target}}} for i in range(5)]


def config_for(folder, data, output):
    config = options(folder)
    config.update(architecture="qwen", qwen_execution="shared_state", method="sft",
        finetuning_type="cl_moe", cl_moe={"experts": 2, "top_k": 1, "max_tasks": 10},
        lora_rank=4, lora_alpha=8, data=str(data), output=str(output), epochs=1,
        global_batch_size=2, lr_scheduler="cosine", warmup_ratio=.2,
        tokens_per_step=8192, microbatch_size=1, microbatch_max_tokens=4096,
        save_every=1, weight_decay=0., lora_lr=.001)
    return config


def test_real_three_primitive_sequence_training_evaluation_and_resume(published, tmp_path, monkeypatch):
    folder, _ = published
    from PIL import Image
    from transformers import AutoProcessor
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
    processor.chat_template = (
        "{% for m in messages %}<|im_start|>{{m['role']}}\n"
        "{% for c in m['content'] %}{% if c['type']=='text' %}{{c['text']}}"
        "{% elif c['type']=='image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% endif %}{% endfor %}<|im_end|>\n{% endfor %}")
    processor.save_pretrained(folder)
    tasks = []
    for name, kind in (("recognition", "choice"), ("judge", "noul"), ("count", "score")):
        data = tmp_path / f"{name}.jsonl"
        samples = records(kind)
        for record in samples:
            record["request"]["state"] = {"messages": [{"role": "user", "content": [
                {"type": "text", "text": "cat"},
                {"type": "image_url", "image_url": {"url": str(image)}}]}]}
        data.write_text("".join(json.dumps(record) + "\n" for record in samples))
        tasks.append({"name": name, "train": data.name, "test": data.name,
                      "train_sha256": file_sha256(data), "test_sha256": file_sha256(data)})
    sequence = tmp_path / "sequence.json"
    sequence.write_text(json.dumps({"format": "coin_decision_v1", "benchmark": BENCHMARK,
        "tasks": tasks, "smoke_limit": 5, "primitives": "typed", "warning": "synthetic CPU test"}))
    config = config_for(folder, "overridden", "overridden")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "run"
    matrix = run_sequence(path, sequence, output)
    assert len(matrix) == 3 and all("accuracy" in cell for cell in matrix[-1])
    calibration = json.loads((output / "calibration_matrix.json").read_text())
    assert calibration["completed_cells"] == calibration["expected_cells"] == 6
    for index, kind in enumerate(("choice", "noul", "score")):
        cell = calibration["cells"][-1][index]
        assert cell["overall"]["count"] == 5
        assert 0 <= cell["overall"]["ece"] <= 1
        assert cell["by_primitive"][kind] == cell["overall"]
        assert cell["overall"]["accuracy"] == matrix[-1][index]["accuracy"]
    checkpoint = torch.load(output / "02_count/consolidated/checkpoint.pt", weights_only=False)
    assert checkpoint["adaptation_buffers"]["cl_moe_state.completed_tasks"] == 3
    for index, name in enumerate(("recognition", "judge", "count")):
        stage = output / f"{index:02d}_{name}"
        rows = [json.loads(line) for line in (stage / "metrics.jsonl").read_text().splitlines()]
        assert [row["states"] for row in rows] == [2, 2, 1]
        assert rows[0]["learning_rates"]["lora"] == 0
        assert rows[1]["learning_rates"]["lora"] == .001
        assert rows[2]["learning_rates"]["lora"] == pytest.approx(.0005)
    monkeypatch.setattr("valen.continual.run_command", lambda *args: pytest.fail("Unexpected rerun"))
    assert run_sequence(path, sequence, output, resume=True) == matrix
    assert json.loads((output / "calibration_matrix.json").read_text()) == calibration
    from evaluation.vqa_cl.calibration import analyze
    assert analyze(sequence, output) == calibration["cells"]


def test_real_cosine_resume_matches_uninterrupted_weights(published, tmp_path, monkeypatch):
    folder, _ = published
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records("choice")))
    config = config_for(folder, data, tmp_path / "continuous")
    run(config)
    interrupted = dict(config, output=str(tmp_path / "interrupted"))
    import valen.training.runner as runner
    original = runner.save_checkpoint

    class Interrupted(Exception):
        pass

    def stop_after_save(*args, **kwargs):
        original(*args, **kwargs)
        raise Interrupted()

    with monkeypatch.context() as patch:
        patch.setattr(runner, "save_checkpoint", stop_after_save)
        with pytest.raises(Interrupted):
            run(interrupted)
    run(interrupted, resume=tmp_path / "interrupted/latest")
    expected = torch.load(tmp_path / "continuous/consolidated/checkpoint.pt", weights_only=False)
    actual = torch.load(tmp_path / "interrupted/consolidated/checkpoint.pt", weights_only=False)
    for name, value in expected["weights"].items():
        torch.testing.assert_close(value, actual["weights"][name], rtol=0, atol=0)
    for name, value in expected["adaptation_buffers"].items():
        torch.testing.assert_close(value, actual["adaptation_buffers"][name], rtol=0, atol=0)


def test_fixed_batch_cosine_two_ranks_with_idle_tail(published, tmp_path):
    import os
    import subprocess
    import sys
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    folder, _ = published
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records("score")))
    config = config_for(folder, data, tmp_path / "distributed")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    worker = tmp_path / "worker.py"
    worker.write_text('''import json, os, sys
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from valen.training.runner import run
def work(rank, config_path):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank))
    dist.init_process_group("gloo", init_method=f"file://{Path(config_path).parent / 'rendezvous'}", rank=rank, world_size=2)
    try:
        run(json.loads(Path(config_path).read_text()))
    finally:
        dist.destroy_process_group()
if __name__ == "__main__":
    mp.spawn(work, args=(sys.argv[1],), nprocs=2, join=True)
''')
    result = subprocess.run([sys.executable, str(worker), str(path)],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                 OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false"),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    logs = [json.loads(line) for line in (tmp_path / "distributed/metrics.jsonl").read_text().splitlines()]
    assert [row["states"] for row in logs] == [2, 2, 1]
    assert [row["learning_rates"]["lora"] for row in logs] == pytest.approx([0, .001, .0005])
    assert [r["states"] for r in logs[-1]["ranks"]] == [1, 0]
    report = json.loads((tmp_path / "distributed/consolidation.json").read_text())
    assert all(layer["observations"] == 5 for layer in report["layers"].values())
