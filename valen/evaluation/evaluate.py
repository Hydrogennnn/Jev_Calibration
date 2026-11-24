"""Evaluate labeled decisions once, sharded without padding across GPUs."""
import argparse
import json
from pathlib import Path
import time

import torch

from valen.training.checkpoint import file_sha256, load_checkpoint
from valen.training.distributed import initialize, close
from valen.evaluation.inference import answer
from valen.modeling.factory import build_model, build_compiler, get_backend, normalize_model_config
from valen.data.schema import read_jsonl
from .metrics import question_metrics, summarize
from valen.modeling.qwen.attention import ATTENTION_IMPLEMENTATIONS, with_attention_implementation


class EvaluationSession:
    """One checkpoint and compiler reused across several evaluation datasets."""

    def __init__(self, checkpoint, context, attn_implementation=None):
        started = time.monotonic()
        self.context, self.checkpoint = context, Path(checkpoint)
        config = json.loads((self.checkpoint / "config.json").read_text(encoding="utf-8"))
        config = with_attention_implementation(normalize_model_config(config), attn_implementation)
        config.update(device=context.device, gradient_checkpointing=False)
        torch.manual_seed(config.get("seed", 42))
        self.config = config
        self.model = build_model(config)
        # payload = load_checkpoint(self.checkpoint, self.model)
        # Do not retain the checkpoint's optimizer tensors for the whole stage.
        # self.progress = dict(payload["progress"])
        self.progress = {"step": None, "epoch": None}
        # del payload
        self.model.eval()
        self.backend = get_backend(config["architecture"])
        self.compiler = build_compiler(config, ".")
        self.checkpoint_sha256 = file_sha256(self.checkpoint / "checkpoint.pt")
        self.load_seconds = time.monotonic() - started
        self.evaluations = 0


@torch.no_grad()
def run(checkpoint, data, output, device="cuda", attn_implementation=None):
    context = initialize(device)
    session = EvaluationSession(checkpoint, context, attn_implementation)
    return evaluate_loaded(session, data, output)


@torch.no_grad()
def evaluate_loaded(session, data, output):
    """Evaluate with an already loaded model; never rebuild or reload weights."""
    run_started = time.monotonic() - (session.load_seconds if session.evaluations == 0 else 0)
    context, checkpoint, config = session.context, session.checkpoint, session.config
    model, backend, compiler = session.model, session.backend, session.compiler
    output, data = Path(output), Path(data)
    output.mkdir(parents=True, exist_ok=True)
    # Both supported compilers resolve relative media against this directory.
    # Switch it on every dataset; retain the loaded processor/tokenizer.
    compiler.media_root = data.parent
    from valen.data.schema import target_distribution
    candidate_fn = backend.candidates
    records = read_jsonl(data, candidate_fn=candidate_fn)
    indices = list(range(context.rank, len(records), context.world_size))
    context.barrier()
    rows, started = [], time.monotonic()
    def synchronize():
        if context.device.startswith("cuda"):
            torch.cuda.synchronize()
    path = output / f"predictions.rank{context.rank}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for local_index, index in enumerate(indices):
            record = records[index]
            compile_started = time.monotonic()
            compiled = compiler.compile(record, labeled_only=True)
            compile_seconds = time.monotonic() - compile_started
            units = iter(backend.inference_units(model, compiled))
            while True:
                synchronize()
                forward_started = time.monotonic()
                unit = next(units, None)
                synchronize()
                if unit is None:
                    break
                unit_seconds = time.monotonic() - forward_started
                forward_seconds = unit_seconds / len(unit)
                for decision in unit:
                    question, logits = decision.question, decision.logits
                    meta = record.get("meta", {})
                    row = {"record_index": index, "record_id": meta.get("record_id"),
                           "group_id": record["group_id"], "qid": question.qid, "task": question.kind,
                           "modality": meta.get("modality", "text" if question.is_text else "media"),
                           "language": meta.get("language_bucket", "unknown"),
                           "domain": meta.get("domain", "unknown"),
                           "timing": {"compile_seconds": compile_seconds, "forward_seconds": forward_seconds,
                                      "compile_plus_forward_seconds": compile_seconds + forward_seconds},
                           "metrics": question_metrics(question, logits)}
                    if backend.unit_scope == "state" or getattr(compiled, "inputs", None) is not None:
                        row["timing"].update(shared_state_forward_seconds=unit_seconds,
                                             forward_scope="amortized_per_question")
                    rows.append(row)
                    probabilities = logits.float().softmax(-1).cpu().tolist()
                    prediction = dict(row, target=dict(zip(question.keys, question.target)),
                                      probabilities=dict(zip(question.keys, probabilities)),
                                      answer=answer(question, logits))
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            if (local_index + 1) % 50 == 0 or local_index + 1 == len(indices):
                stream.flush()
                print(json.dumps({"event": "evaluation_progress", "rank": context.rank,
                                  "records": local_index + 1, "total": len(indices),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    rank_timing = context.gather({"rank": context.rank, "records": len(indices),
                                 "evaluation_seconds": time.monotonic() - started,
                                 "load_and_evaluation_seconds": time.monotonic() - run_started,
                                 "device": torch.cuda.get_device_name() if context.device.startswith("cuda") else "cpu"})
    gathered = context.gather(rows)
    report = None
    if context.primary:
        all_rows = [row for rank_rows in gathered for row in rank_rows]
        # Check coverage without recompiling media.
        expected = {(i, qid) for i, record in enumerate(records)
                    for qid, q in record["request"]["questions"].items()
                    if target_distribution(record.get("targets", {}).get(qid), [k for k, _ in candidate_fn(q)]) is not None}
        actual = [(row["record_index"], row["qid"]) for row in all_rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Evaluation duplicated or omitted labeled questions")
        report = {"checkpoint": str(checkpoint), "training_step": session.progress["step"],
                  "base_model_path": backend.base_paths(config), "seed": config.get("seed", 42),
                  "architecture": config.get("architecture", "qwen"),
                  "training_epoch": session.progress["epoch"], "data": str(data),
                  "data_sha256": file_sha256(data),
                  "checkpoint_sha256": session.checkpoint_sha256,
                  "model_reused": session.evaluations > 0,
                  "records": len(records), "questions": len(all_rows), "world_size": context.world_size,
                  "temperature": 1.0, "elapsed_seconds": time.monotonic() - started,
                  "rank_timing": rank_timing,
                  "evaluation_wall_seconds": max(r["evaluation_seconds"] for r in rank_timing),
                  "load_and_evaluation_wall_seconds": max(r["load_and_evaluation_seconds"] for r in rank_timing),
                  "metrics": summarize(all_rows)}
        if hasattr(model, "attention_backends"):
            report["attention_backends"] = model.attention_backends
        report["questions_per_second"] = len(all_rows) / report["evaluation_wall_seconds"]
        report["latency_seconds"] = {}
        for name in ("compile_seconds", "forward_seconds", "compile_plus_forward_seconds"):
            values = torch.tensor([row["timing"][name] for row in all_rows], dtype=torch.float64)
            report["latency_seconds"][name] = {"mean": float(values.mean()), "p50": float(values.quantile(.5)),
                                               "p95": float(values.quantile(.95))}
        (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        predictions = []
        for rank in range(context.world_size):
            with (output / f"predictions.rank{rank}.jsonl").open(encoding="utf-8") as stream:
                predictions.extend(json.loads(line) for line in stream)
        predictions.sort(key=lambda row: (row["record_index"], row["qid"]))
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "evaluation_complete", "questions": len(all_rows),
                          "metrics_path": str(output / "metrics.json")}), flush=True)
    context.barrier()
    session.evaluations += 1
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", choices=ATTENTION_IMPLEMENTATIONS,
                        help="Override the Qwen checkpoint's attention backend")
    args = parser.parse_args()
    try:
        run(args.checkpoint, args.data, args.output, args.device, args.attn_implementation)
    finally:
        close()


if __name__ == "__main__":
    main()
