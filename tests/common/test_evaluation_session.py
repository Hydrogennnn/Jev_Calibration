"""Test real session/worker control flow with lightweight model dependencies."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class EvaluationSessionTests(unittest.TestCase):
    def test_model_weights_and_processor_load_once_media_root_changes_per_dataset(self):
        torch = ModuleType("torch")
        torch.no_grad = lambda: lambda function: function
        torch.manual_seed = Mock()
        checkpoint = ModuleType("valen.training.checkpoint")
        checkpoint.file_sha256 = Mock(return_value="checkpoint-hash")
        checkpoint.load_checkpoint = Mock(return_value={"progress": {"step": 3, "epoch": 1}})
        distributed = ModuleType("valen.training.distributed")
        distributed.initialize = Mock()
        distributed.close = Mock()
        inference = ModuleType("valen.evaluation.inference")
        inference.answer = Mock()
        factory = ModuleType("valen.modeling.factory")
        factory.normalize_model_config = lambda config: config
        factory.build_model = Mock(return_value=SimpleNamespace(eval=Mock()))
        roots = []
        compiler = SimpleNamespace(media_root=Path("."))
        def compile_record(*args, **kwargs):
            roots.append(compiler.media_root)
            return SimpleNamespace()
        compiler.compile = compile_record
        factory.build_compiler = Mock(return_value=compiler)
        backend = SimpleNamespace(candidates=Mock(), inference_units=lambda *args: iter(()))
        factory.get_backend = Mock(return_value=backend)
        metrics = ModuleType("valen.evaluation.metrics")
        metrics.question_metrics = Mock()
        metrics.summarize = Mock()
        attention = ModuleType("valen.modeling.qwen.attention")
        attention.ATTENTION_IMPLEMENTATIONS = ("sdpa",)
        attention.with_attention_implementation = lambda config, attention: config
        modules = (torch, checkpoint, distributed, inference, factory, metrics, attention)
        path = Path(__file__).resolve().parents[2] / "valen/evaluation/evaluate.py"
        spec = importlib.util.spec_from_file_location("valen.evaluation._session_under_test", path)
        worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {module.__name__: module for module in modules}):
            spec.loader.exec_module(worker)
        # Non-primary CPU worker: check loading/compilation reuse independently
        # of numeric metrics and distributed aggregation tested elsewhere.
        context = SimpleNamespace(rank=0, world_size=1, primary=False, device="cpu",
                                  barrier=lambda: None, gather=lambda value: [value])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"architecture": "qwen"}))
            worker.read_jsonl = Mock(return_value=[{"request": {}, "group_id": "one"}])
            session = worker.EvaluationSession(root, context)
            for task in ("first", "second"):
                worker.evaluate_loaded(session, root / task / "test.jsonl", root / task / "evaluation")
            factory.build_model.assert_called_once()
            checkpoint.load_checkpoint.assert_called_once()
            factory.build_compiler.assert_called_once()
            self.assertEqual(roots, [root / "first", root / "second"])
            self.assertEqual(session.evaluations, 2)
            self.assertEqual(session.progress["step"], 3)


if __name__ == "__main__":
    unittest.main()
