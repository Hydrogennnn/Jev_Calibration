"""Convert the published CL4VQA splits into reproducible Valen decisions.

This is a candidate-based adaptation, not the authors' generative VQA protocol.
Task membership comes exclusively from the published filenames, never heuristics.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from valen.data.schema import validate_record
from .protocol import ABBREVIATIONS, BENCHMARK, DATA_URL, REFERENCE_REVISION, REFERENCE_URL, TASKS, task_order


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer_key(answer):
    # The reference evaluator uses uppercase equality. Keep punctuation and the
    # displayed spelling; never turn a label such as "." into an empty string.
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Expected one nonempty string answer, got {answer!r}")
    return answer.strip().upper()


def image_path(root, value, allow_missing):
    if not isinstance(value, str) or not value:
        raise ValueError("CL4VQA requires an image path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a relative COCO image path: {value}")
    # Authors use ./COCO2014/train2014/..., whereas the CoIN downloader uses
    # COCO/train2014/.... Accept either parent root or the COCO directory itself.
    suffix = Path(*relative.parts[1:]) if relative.parts[0] in {"COCO2014", "COCO"} else relative
    paths = list(dict.fromkeys((root / relative, root / suffix,
                               root / "COCO2014" / suffix, root / "COCO" / suffix)))
    found = next((path for path in paths if path.is_file()), None)
    if found is None and not allow_missing:
        raise FileNotFoundError(f"Missing image {value}; checked: {', '.join(map(str, paths))}")
    return str((found or paths[0]).resolve())


def read_samples(path, task, root, *, limit=None, allow_missing=False):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON array")
    samples = []
    resolved = {}
    for row_index, row in enumerate(rows):
        try:
            identifier = row.get("question_id", row.get("id"))
            if identifier is None or str(identifier) == "":
                raise ValueError("Missing question_id/id")
            if "conversations" in row:
                turns = row["conversations"]
                if not turns or len(turns) % 2:
                    raise ValueError("Expected complete human/assistant pairs")
                pairs = []
                for i in range(0, len(turns), 2):
                    if turns[i]["from"] != "human" or turns[i + 1]["from"] not in {"gpt", "assistant"}:
                        raise ValueError("Expected alternating human/assistant turns")
                    pairs.append((turns[i]["value"], turns[i + 1]["value"]))
            else:
                pairs = [(row["text"], row["answer"])]
            image = row["image"]
            if image not in resolved:
                resolved[image] = image_path(root, image, allow_missing)
            for turn, (text, answer) in enumerate(pairs):
                answer_key(answer)
                text = text.replace("<image>", "").strip()
                text = re.sub(r"\s*Answer the question using a single word or phrase\.?\s*$", "", text).strip()
                if not text:
                    raise ValueError("Empty question")
                samples.append({"id": f"{task}:{identifier}:{row_index}:{turn}",
                    "source_id": str(identifier), "source_row": row_index, "turn": turn,
                    "group": hashlib.sha256(Path(image).name.encode()).hexdigest(),
                    "image": resolved[image], "text": text, "answer": answer.strip()})
                if limit and len(samples) >= limit:
                    return samples, len(rows)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:row {row_index}: {exc}") from exc
    return samples, len(rows)


class AnswerPool:
    def __init__(self, samples):
        self.display = {}
        for sample in samples:
            self.display.setdefault(answer_key(sample["answer"]), sample["answer"])
        self.keys = sorted(self.display)

    def choices(self, gold, count, rng):
        gold_key = answer_key(gold)
        # Map sampled indices around the gold's position without copying the
        # whole vocabulary on every row. Never import test or future-task labels.
        from bisect import bisect_left
        position = bisect_left(self.keys, gold_key)
        present = position < len(self.keys) and self.keys[position] == gold_key
        size = len(self.keys) - int(present)
        indices = rng.sample(range(size), min(count - 1, size))
        values = [gold] + [self.display[self.keys[i + int(present and i >= position)]] for i in indices]
        if len(values) < 2:
            raise ValueError("Training answer pool cannot supply a distinct distractor")
        return values


def convert(sample, task, pool, choices, seed, split, primitives="typed", count_max=20):
    rng = random.Random(int(hashlib.sha256(f"{seed}:{split}:{sample['id']}".encode()).hexdigest()[:16], 16))
    answer, key = sample["answer"], answer_key(sample["answer"])
    question = {"instructions": sample["text"]}
    fallback = None
    if primitives == "typed" and task == "judge" and key in {"YES", "NO"}:
        question["type"] = "noul"
        mapping = {"true": "yes", "false": "no"}
        target = {"true": float(key == "YES"), "false": float(key == "NO")}
    elif (primitives == "typed" and task == "count" and re.fullmatch(r"0|[1-9][0-9]*", answer)
          and int(answer) <= count_max):
        question.update(type="score", criteria=[str(i) for i in range(count_max + 1)])
        mapping = {str(i): str(i) for i in range(count_max + 1)}
        target = {name: float(value == answer) for name, value in mapping.items()}
    else:
        if primitives == "typed" and task in {"judge", "count"}:
            fallback = "non_binary_answer" if task == "judge" else "outside_count_scale_or_noncanonical_integer"
        values = pool.choices(answer, choices, rng)
        rng.shuffle(values)
        mapping = {f"option_{i}": value for i, value in enumerate(values)}
        target = {name: float(answer_key(value) == key) for name, value in mapping.items()}
        question.update(type="choice", criteria=mapping)
    record = {"group_id": sample["group"],
        "request": {"state": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Use the image to answer the question."},
            {"type": "image_url", "image_url": {"url": sample["image"]}}]}]},
            "questions": {"q0": question}},
        "targets": {"q0": {"probabilities": target}},
        "meta": {"record_id": sample["id"], "source_id": sample["source_id"],
            "source_row": sample["source_row"], "turn": sample["turn"], "source": "CL4VQA",
            "domain": task, "split": split, "modality": "image", "language_bucket": "en",
            "protocol": BENCHMARK, "gold_answer": answer, "answer_map": mapping,
            "candidate_count": len(mapping), "primitive_fallback": fallback}}
    return validate_record(record)


def prepare(source, image_root, output, tasks=TASKS, choices=8, seed=42, limit=None,
            allow_missing_images=False, primitives="typed", count_max=20):
    tasks = task_order(tasks)
    if type(choices) is not int or not 2 <= choices <= 255:
        raise ValueError("choices must be in [2, 255]")
    if type(count_max) is not int or not 1 <= count_max <= 254 or primitives not in {"typed", "choice"}:
        raise ValueError("Require primitives=typed/choice and count_max in [1, 254]")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    source, image_root, output = map(lambda p: Path(p).resolve(), (source, image_root, output))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    # Reuse the existing sequence container and runner; benchmark identifies the
    # distinct data/protocol, despite the historical name of the container format.
    manifest = {"format": "coin_decision_v1", "benchmark": BENCHMARK,
        "reference": {"repository": REFERENCE_URL, "revision": REFERENCE_REVISION, "data": DATA_URL},
        "seed": seed, "choices": choices, "primitives": primitives, "count_max": count_max,
        "smoke_limit": limit, "image_root": str(image_root), "images_checked": not allow_missing_images,
        "eval_source": "authors_released_test", "tasks": [],
        "warning": "Valen candidate adaptation: not generative CL-MoE accuracy or official VQA consensus scoring."}
    groups = {"train": set(), "test": set()}
    try:
        for task in tasks:
            print(f"Processing {task} ...", flush=True)
            paths = {split: source / split / f"{split}_q_{task}.json" for split in groups}
            samples, original_counts = {}, {}
            for split, path in paths.items():
                samples[split], original_counts[split] = read_samples(path, task, image_root,
                    limit=limit, allow_missing=allow_missing_images)
            pool = AnswerPool(samples["train"])
            folder = staging / task
            folder.mkdir()
            kinds, fallbacks = {}, {}
            for split, rows in samples.items():
                kinds[split], fallbacks[split] = Counter(), Counter()
                with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                    for sample in rows:
                        try:
                            record = convert(sample, task, pool, choices, seed, split, primitives, count_max)
                        except (ValueError, TypeError, KeyError) as exc:
                            raise ValueError(f"{task}/{split}/{sample['id']}: {exc}") from exc
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        kinds[split][record["request"]["questions"]["q0"]["type"]] += 1
                        if record["meta"]["primitive_fallback"]:
                            fallbacks[split][record["meta"]["primitive_fallback"]] += 1
                        groups[split].add(record["group_id"])
            (folder / "answer_pool.json").write_text(json.dumps(pool.display, ensure_ascii=False) + "\n")
            manifest["tasks"].append({"name": task, "abbreviation": ABBREVIATIONS[TASKS.index(task)],
                "train": f"{task}/train.jsonl", "test": f"{task}/test.jsonl",
                "counts": {s: len(v) for s, v in samples.items()}, "source_rows": original_counts,
                "source_sha256": {str(p): digest(p) for p in paths.values()},
                "train_sha256": digest(folder / "train.jsonl"), "test_sha256": digest(folder / "test.jsonl"),
                "primitive_counts": kinds, "fallback_counts": fallbacks})
        manifest["cross_task_train_eval_overlapping_groups"] = len(groups["train"] & groups["test"])
        (staging / "sequence.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/CL4VQA")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="data/vqa-cl-decision")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primitives", choices=("typed", "choice"), default="typed")
    parser.add_argument("--count-max", type=int, default=20, help="Fixed score scale 0..N; other answers use choice")
    parser.add_argument("--limit", type=int, help="Smoke subset: first N expanded QA pairs per task/split")
    parser.add_argument("--allow-missing-images", action="store_true", help="Conversion checks only; training needs real images")
    args = vars(parser.parse_args())
    manifest = prepare(**args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
