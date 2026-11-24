"""Real sequence/calibration I/O with injected model workers (no GPU required)."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import weakref

from valen.evaluation.eval_continual import evaluate_sequence, load_plan, summarize_matrix
from evaluation.vqa_cl.calibration import CalibrationTracker, digest, write_json


class Context:
    rank, world_size, primary = 0, 1, True

    def gather(self, value):
        return [value]

    def barrier(self):
        pass


class Session:
    pass


class ContinualEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "training"
        self.run.mkdir()
        self.sequence = self.root / "sequence.json"
        self.output = self.root / "evaluation"
        self.tasks = ["agnews", "legacy_policy", "compositional"]
        tasks = []
        for i, name in enumerate(self.tasks):
            data = self.root / f"{name}.jsonl"
            rows = []
            for k, kind in enumerate(("choice", "noul", "score")):
                criteria = {"a": "A", "b": "B"} if kind == "choice" else ["low", "high"]
                q = {"type": kind, "instructions": "Decide"}
                if kind != "noul":
                    q["criteria"] = criteria
                keys = {"choice": ["a", "b"], "noul": ["true", "false"], "score": ["0", "1"]}[kind]
                rows.append({"group_id": f"{name}-{k}", "meta": {"record_id": f"{name}-{k}"},
                    "request": {"state": "Text", "questions": {"q": q}},
                    "targets": {"q": {"probabilities": {keys[0]: 1., keys[1]: 0.}}}})
            data.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tasks.append({"name": name, "test": data.name, "test_sha256": digest(data)})
            stage = self.run / f"{i:02d}_{name}"
            checkpoint = stage / "latest"
            checkpoint.mkdir(parents=True)
            config = {"finetuning_type": "lora"}
            write_json(stage / "stage_config.json", config)
            write_json(checkpoint / "config.json", config)
            (checkpoint / "checkpoint.pt").write_bytes(f"checkpoint-{i}".encode())
            write_json(stage / "stage_complete.json", {"checkpoint_sha256": digest(checkpoint / "checkpoint.pt")})
        write_json(self.sequence, {"format": "coin_decision_v1", "benchmark": "decision_v7_source_cl_v1", "tasks": tasks})
        write_json(self.run / "sequence_run.json", {"sequence_sha256": digest(self.sequence), "tasks": self.tasks})
        self.plan = load_plan(self.sequence, self.run)
        self.loads, self.evaluations, self.references = [], [], []
        self.training_bytes = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}

    def load(self, checkpoint, context, attention):
        self.assertTrue(all(ref() is None for ref in self.references), "Previous model was not released")
        session = Session()
        session.checkpoint = checkpoint
        self.references.append(weakref.ref(session))
        self.loads.append(checkpoint)
        return session

    def evaluate(self, session, data, output):
        self.evaluations.append((session.checkpoint, data))
        output.mkdir(parents=True, exist_ok=True)
        predictions = []
        for i, row in enumerate(map(json.loads, Path(data).read_text().splitlines())):
            target = row["targets"]["q"]["probabilities"]
            keys = list(target)
            probs = dict(zip(keys, [.2, .8] if i == 1 else [.8, .2]))
            predictions.append({"record_index": i, "qid": "q", "record_id": row["meta"]["record_id"],
                "group_id": row["group_id"], "task": row["request"]["questions"]["q"]["type"],
                "target": target, "probabilities": probs})
        (output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
        write_json(output / "metrics.json", {"checkpoint_sha256": digest(Path(session.checkpoint) / "checkpoint.pt"),
            "data_sha256": digest(data), "temperature": 1.,
            "metrics": {"overall": {"accuracy": 2/3, "nll": .6, "brier": .4}}})

    def execute(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return evaluate_sequence(self.plan, self.output, Context(), self.load, self.evaluate,
                                     lambda: None, **kwargs)

    def test_t_loads_triangular_evaluations_calibration_and_release(self):
        matrix = self.execute()
        self.assertEqual(len(self.loads), 3)
        self.assertEqual(len(self.evaluations), 6)
        self.assertTrue(all(ref() is None for ref in self.references))
        self.assertEqual([sum(cell is not None for cell in row) for row in matrix], [1, 2, 3])
        summary = json.loads((self.output / "continual_metrics.json").read_text())
        self.assertEqual(summary["model_loads_per_rank_this_run"], 3)
        self.assertAlmostEqual(summary["average_performance"], 2/3)
        self.assertEqual(summary["backward_transfer"], 0)
        calibration = json.loads((self.output / "calibration_matrix.json").read_text())
        self.assertEqual(calibration["completed_cells"], 6)
        self.assertAlmostEqual(calibration["ece_matrix"][2][0], .8 - 2/3)
        self.assertTrue((self.output / "calibration_tasks.csv").is_file())
        for p, data in self.training_bytes.items():
            self.assertEqual(p.read_bytes(), data)
        # The VQA-CL entry still rejects unrelated benchmark/task names by default.
        with self.assertRaisesRegex(ValueError, "CL4VQA"):
            CalibrationTracker(self.sequence, self.output)

    def test_resume_skips_loaded_models_and_recomputes_only_damaged_cell(self):
        self.execute()
        self.loads.clear()
        self.evaluations.clear()
        self.execute(resume=True)
        self.assertFalse(self.loads)
        self.assertFalse(self.evaluations)
        destination = self.output / "02_compositional/eval/agnews/predictions.jsonl"
        destination.write_text("damaged\n")
        self.execute(resume=True)
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(len(self.evaluations), 1)
        self.assertIn("02_compositional", self.loads[0])

    def test_failure_releases_model_and_leaves_cell_unsealed(self):
        def fail(*args):
            raise RuntimeError("inference failed")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "inference failed"):
            evaluate_sequence(self.plan, self.output, Context(), self.load, fail, lambda: None)
        self.assertTrue(all(ref() is None for ref in self.references))
        self.assertFalse(list(self.output.rglob("evaluation_complete.json")))

    def test_preflight_validates_data_checkpoints_and_completion(self):
        data = self.root / "agnews.jsonl"
        original = data.read_bytes()
        data.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "data changed"):
            load_plan(self.sequence, self.run)
        data.write_bytes(original)
        checkpoint = self.run / "02_compositional/latest/checkpoint.pt"
        checkpoint.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            load_plan(self.sequence, self.run)

    def test_moe_uses_consolidated_and_fails_without_it(self):
        stage = self.run / "01_legacy_policy"
        write_json(stage / "stage_config.json", {"model": {"finetuning_type": "cl_moe"}})
        with self.assertRaises(FileNotFoundError):
            load_plan(self.sequence, self.run)
        (stage / "latest").rename(stage / "consolidated")
        plan = load_plan(self.sequence, self.run)
        self.assertTrue(plan["stages"][1]["checkpoint"].endswith("consolidated"))

    def test_output_guard_and_partial_summary(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            evaluate_sequence(self.plan, self.run, Context(), self.load, self.evaluate, lambda: None)
        self.execute(calibration=False)
        self.assertFalse((self.output / "calibration_matrix.json").exists())
        with self.assertRaisesRegex(ValueError, "output exists"):
            self.execute(calibration=False)
        result = summarize_matrix([[{"accuracy": .8}, None], [None, None]])
        self.assertIsNone(result["average_performance"])
        self.assertEqual(result["average_performance_by_stage"], [.8, None])
        result = summarize_matrix([[{"accuracy": .8}, None], [{"accuracy": .5}, {"accuracy": .9}]])
        self.assertAlmostEqual(result["backward_transfer"], -.3)
        self.assertAlmostEqual(result["average_forgetting"], .3)

    def test_soft_labels_fail_calibration_before_loading_any_model(self):
        data = self.root / "agnews.jsonl"
        rows = list(map(json.loads, data.read_text().splitlines()))
        rows[0]["targets"]["q"]["probabilities"] = {"a": .4, "b": .6}
        data.write_text("".join(json.dumps(row) + "\n" for row in rows))
        sequence = json.loads(self.sequence.read_text())
        sequence["tasks"][0]["test_sha256"] = digest(data)
        write_json(self.sequence, sequence)
        write_json(self.run / "sequence_run.json", {"sequence_sha256": digest(self.sequence), "tasks": self.tasks})
        self.plan = load_plan(self.sequence, self.run)
        with self.assertRaisesRegex(ValueError, "hard one-hot"):
            self.execute()
        self.assertFalse(self.loads)


if __name__ == "__main__":
    unittest.main()
