"""Run without training dependencies: python3 -m unittest discover -s tests/integration -p test_decision_v7.py."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.decision_v7.prepare import SPLITS, convert, digest, prepare
from valen.data.schema import read_jsonl


def sample(source="public", split="train", index=0):
    identifier = f"{source}/{split}/{index}"
    return {"state": {"document": "A review", "context": ["one", "two"]},
        "questions": {
            "topic": {"type": "choice", "instructions": "Choose a category", "src": "public_choice",
                      "criteria": {"a": None, "b": {"what": "Category B"}}, "label": "b"},
            "rating": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"],
                       "label": 0, "src": "public_score"},
            "positive": {"type": "noul", "instructions": "Positive?", "label": False,
                         "criteria": {"true": "Positive overall", "false": "Negative or mixed"}}},
        "_meta": {"source": source, "id": identifier, "group_id": identifier,
                  "certificate": {"label": "must stay outside the prompt"}}}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_source(root):
    root.mkdir()
    for split in SPLITS:
        sources = ["public", "synthetic"] if split != "test" else ["public", "eval_only"]
        write_rows(root / f"{split}.jsonl", [sample(source, split) for source in sources])
    (root / "manifest.json").write_text(json.dumps({"trainable_sources": ["synthetic", "public"]}))


class DecisionV7Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        make_source(self.source)

    def test_labels_descriptions_rubrics_and_structured_text(self):
        raw = sample()
        before = deepcopy(raw)
        record = convert(raw, "train", 1)
        self.assertEqual(raw, before)
        self.assertEqual(json.loads(record["request"]["state"]), raw["state"])
        questions = record["request"]["questions"]
        self.assertEqual(questions["topic"]["criteria"], {"a": "a", "b": '{"what": "Category B"}'})
        self.assertIn("Negative or mixed", questions["positive"]["instructions"])
        self.assertEqual(record["targets"]["positive"]["probabilities"], {"true": 0., "false": 1.})
        self.assertEqual(record["targets"]["rating"]["probabilities"], {"0": 1., "1": 0.})
        self.assertEqual(record["targets"]["topic"]["probabilities"], {"a": 0., "b": 1.})
        prompt = json.dumps(record["request"])
        for forbidden in ('"label"', '"src"', '"_meta"', '"certificate"'):
            self.assertNotIn(forbidden, prompt)
        raw["state"] = ["paragraph", {"policy": "rule"}]
        self.assertEqual(json.loads(convert(raw, "train", 1)["request"]["state"]), raw["state"])
        raw["questions"]["topic"]["instructions"] = {"question": "Pick", "focus": "Whole message"}
        converted = convert(raw, "train", 1)
        self.assertEqual(json.loads(converted["request"]["questions"]["topic"]["instructions"]),
                         raw["questions"]["topic"]["instructions"])

    def test_invalid_labels_fail_instead_of_coercing_or_dropping(self):
        for qid, label in (("positive", 1), ("rating", True), ("rating", -1),
                           ("rating", 2), ("topic", "missing")):
            with self.subTest(qid=qid, label=label):
                raw = sample()
                raw["questions"][qid]["label"] = label
                with self.assertRaises(ValueError):
                    convert(raw, "train", 1)
        raw = sample()
        del raw["questions"]["topic"]["label"]
        with self.assertRaises(KeyError):
            convert(raw, "train", 1)

    def test_native_splits_task_order_hashes_and_determinism(self):
        a, b = self.root / "a", self.root / "b"
        manifest = prepare(self.source, a)
        self.assertEqual(prepare(self.source, b), manifest)
        self.assertEqual([t["name"] for t in manifest["tasks"]], ["synthetic", "public"])
        self.assertEqual(manifest["eval_only_sources"], ["eval_only"])
        self.assertTrue(all(count == 0 for count in manifest["split_group_overlaps"].values()))
        for task in manifest["tasks"]:
            self.assertEqual(task["test"], f'{task["name"]}/development.jsonl')
            for split in ("train", "test"):
                self.assertEqual(digest(a / task[split]), task[f"{split}_sha256"])
        self.assertTrue((a / "eval_only/test.jsonl").is_file())
        self.assertFalse((a / "synthetic/test.jsonl").exists())
        total = 0
        for name, files in manifest["files"].items():
            for split, info in files.items():
                path = a / info["path"]
                self.assertEqual(path.read_bytes(), (b / info["path"]).read_bytes())
                rows = read_jsonl(path)
                total += len(rows)
                self.assertEqual(len(rows), info["records"])
                self.assertEqual(sum(len(r["targets"]) for r in rows), info["questions"])
                self.assertTrue(all(r["meta"]["source"] == name and r["meta"]["split"] == split for r in rows))
        self.assertEqual(total, 8)
        with self.assertRaises(FileExistsError):
            prepare(self.source, a)

    def test_explicit_test_split_requires_coverage_and_keeps_other_data(self):
        output = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "No test records"):
            prepare(self.source, output, eval_split="test")
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".missing-*")))
        manifest = prepare(self.source, output, tasks=["public"], eval_split="test")
        self.assertEqual(manifest["tasks"][0]["test"], "public/test.jsonl")
        self.assertEqual(manifest["unselected_train_sources"], ["synthetic"])
        self.assertTrue((output / "synthetic/train.jsonl").is_file())
        with self.assertRaisesRegex(ValueError, "unique sources"):
            prepare(self.source, self.root / "invalid", tasks=["eval_only"])

    def test_cross_source_group_leakage_is_rejected_atomically(self):
        row = sample("eval_only", "test")
        row["_meta"]["group_id"] = "public/train/0"
        write_rows(self.source / "test.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "Split group overlap"):
            prepare(self.source, self.root / "leaked")
        self.assertFalse((self.root / "leaked").exists())
        self.assertFalse(list(self.root.glob(".leaked-*")))

    def test_invalid_source_and_duplicate_ids_have_line_context(self):
        for rows, expected in (([sample("../escape")], "Invalid source"),
                               ([sample(), sample()], "Duplicate record ID")):
            write_rows(self.source / "train.jsonl", rows)
            with self.assertRaisesRegex(ValueError, rf"train.jsonl:\d+: .*{expected}"):
                prepare(self.source, self.root / "invalid")
            self.assertFalse((self.root / "invalid").exists())

    def test_input_checksum_is_verified(self):
        path = self.source / "manifest.json"
        data = json.loads(path.read_text())
        data["files"] = {"train.jsonl": {"sha256": "0" * 64}}
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            prepare(self.source, self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
