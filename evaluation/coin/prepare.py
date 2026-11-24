"""Convert official CoIN conversations/evaluation rows to Valen JSONL.

This is a fixed, gold-included multiple-choice adaptation, NOT original CoIN.
No generator or training dependency is required. Test answers never enter the
negative pool. Original VQA test-server files without labels are rejected.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile

from valen.data.schema import validate_record

# TASKS = ("ScienceQA", "TextVQA", "ImageNet", "GQA", "VizWiz", "Grounding", "VQAv2", "OCRVQA")
TASKS = ("ScienceQA", "TextVQA", "GQA", "VizWiz", "Grounding", "VQAv2", "OCRVQA")
ALIASES = {"OCR-VQA": "OCRVQA", "VQA-v2": "VQAv2"}
FORMATTING = re.compile(
    r"(?:Answer (?:the|each) question using a single word or phrase\.?|"
    r"Answer with the option['’]s letter from the given choices directly\.?)", re.I)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def normalize_answer(value):
    text = re.sub(r"\s+", " ", str(value).strip()).casefold()
    # Keep punctuation-only answers such as "." instead of erasing the label.
    return text.strip(". ") or text


def answer_counts(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        if "answer" not in value:
            raise ValueError("Answer objects require an answer field")
        value = value["answer"]
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("Empty answer list")
    counts = Counter(normalize_answer(v["answer"] if isinstance(v, dict) else v) for v in values)
    if "" in counts:
        raise ValueError("Empty answer")
    return dict(counts)


def read_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and not any(k in data for k in ("annotations", "data")):
        return {str(k): v for k, v in data.items()}
    rows = data if isinstance(data, list) else data.get("annotations", data.get("data"))
    result = {}
    for row in rows:
        # VQAv2/TextVQA use question_id; official VizWiz uses image filename.
        key = str(row.get("question_id", row.get("image", row.get("id", ""))))
        if not key or key in result:
            raise ValueError(f"Missing/duplicate annotation ID: {key}")
        result[key] = row.get("answers", row.get("multiple_choice_answer", row.get("answer")))
    return result


def read_samples(path, task, image_root, annotations=None):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array")
    samples = []
    annotations = annotations or {}
    for row_index, row in enumerate(rows):
        identifier = str(row.get("id", row.get("question_id", "")))
        if not identifier:
            raise ValueError(f"{path}: row without id/question_id")
        image = row.get("image")
        # Official VizWiz val annotations contain duplicated .jpg suffixes.
        if task == "VizWiz" and image and image.endswith(".jpg.jpg"):
            image = image[:-4]
        media = str((image_root / image).resolve()) if image else None
        if "conversations" in row:
            turns = row["conversations"]
            if len(turns) % 2:
                raise ValueError(f"{path}:{identifier}: incomplete conversation")
            pairs = []
            for i in range(0, len(turns), 2):
                if turns[i]["from"] != "human" or turns[i + 1]["from"] not in {"gpt", "assistant"}:
                    raise ValueError(f"{path}:{identifier}: expected alternating human/assistant")
                pairs.append((turns[i]["value"], turns[i + 1]["value"]))
        else:
            answer = row.get("answers", row.get("answer_bbox" if task == "Grounding" else "answer"))
            if answer is None:
                answer = annotations.get(identifier)
                if answer is None and image:
                    answer = annotations.get(Path(image).name)
            pairs = [(row["text"], answer)]
        for turn, (question, answer) in enumerate(pairs):
            # Some official train question_id values are image IDs and repeat
            # across distinct QA rows. Preserve them without dropping samples.
            key = f"{task}:{identifier}:{row_index}:{turn}"
            text = FORMATTING.sub("", question.replace("<image>", "")).strip()
            if not text:
                raise ValueError(f"{key}: empty question")
            # Root-independent identity preserves splits when images move hosts.
            group = "media:" + Path(image).as_posix() if image else f"text:{task}:{identifier}"
            samples.append(dict(id=key, group=hashlib.sha256(group.encode()).hexdigest(),
                image=media, text=text, answer=answer, source_id=identifier, source_row=row_index, turn=turn))
    return samples


def science_options(text, answer):
    matches = list(re.finditer(r"(?m)^\s*([A-Z])[.)]\s+", text))
    if len(matches) < 2:
        raise ValueError("ScienceQA requires at least two lettered options")
    choices = [text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
               for i, m in enumerate(matches)]
    letter = str(answer).strip().strip("(). ").upper()
    labels = [m.group(1) for m in matches]
    # if letter not in labels or \
    #     len(set(labels)) != len(labels) or len(set(map(normalize_answer, choices))) != len(choices):
    #     raise ValueError("Invalid ScienceQA label/options")
    if (
        letter not in labels
        or len(set(labels)) != len(labels)
        or len(set(choices)) != len(choices)
    ):
        raise ValueError("Invalid ScienceQA label/options")
    return text[:matches[0].start()].strip(), choices, labels.index(letter)


COLORS = set("red blue green yellow black white brown orange pink purple gray grey silver gold".split())


def answer_type(answer):
    if answer in {"yes", "no"}:
        return "boolean"
    if answer in COLORS:
        return "color"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
        return "number"
    return "text"


class AnswerPool:
    def __init__(self, samples):
        answers = set()
        for sample in samples:
            try:
                answers.update(answer_counts(sample["answer"]))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"{sample.get('id', '<unknown>')}: invalid training answer "
                    f"{sample.get('answer')!r}: {exc}"
                ) from exc
        self.all = sorted(answers)
        self.by_type = defaultdict(list)
        for answer in self.all:
            self.by_type[answer_type(answer)].append(answer)

    def choices(self, counts, k, rng):
        positives = sorted(counts)
        if len(positives) > k:
            raise ValueError("More acceptable answers than candidates; increase --choices")
        if set(positives) <= {"yes", "no"}:
            return ["yes", "no"]
        selected = list(positives)
        if len(selected) == k:
            return selected
        for pool in (self.by_type[answer_type(positives[0])], self.all):
            # Bounded sampling avoids an O(dataset * answer_vocabulary) pass.
            for value in rng.sample(pool, min(len(pool), k * 4)):
                if value not in selected:
                    selected.append(value)
                if len(selected) == k:
                    return selected
        if len(selected) < 2:
            raise ValueError("Insufficient training answers to construct distractors")
        return selected


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda v: (v[2] - v[0]) * (v[3] - v[1])
    return intersection / (area(a) + area(b) - intersection)


def parse_box(value):
    box = json.loads(value) if isinstance(value, str) else value
    if (not isinstance(box, list) or len(box) != 4 or
        any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
        not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
        raise ValueError("Grounding requires a nonempty normalized xyxy box")
    return list(map(float, box))


def grounding_choices(answer, k, rng):
    gold = parse_box(answer)
    boxes = [gold]
    for _ in range(k * 200):
        x1, x2 = sorted([rng.random(), rng.random()])
        y1, y2 = sorted([rng.random(), rng.random()])
        box = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
        if x2 - x1 > 0.03 and y2 - y1 > 0.03 and box_iou(gold, box) < 0.3 and box not in boxes:
            boxes.append(box)
        if len(boxes) == k:
            return boxes
    raise ValueError("Could not generate distinct negative boxes")


def convert(sample, task, pool, k, seed, split):
    if sample["answer"] is None:
        raise ValueError(f"{sample['id']}: missing labels; use labeled val plus --annotations, or --eval-source holdout")
    rng = random.Random(stable_seed(seed, task, sample["id"], split))
    text = sample["text"]
    extra = {}
    if task == "ScienceQA":
        text, choices, correct = science_options(text, sample["answer"])
        weights = [float(i == correct) for i in range(len(choices))]
        protocol = "native_multiple_choice"
    elif task == "Grounding":
        boxes = grounding_choices(sample["answer"], k, rng)
        choices = ["Bounding box (normalized x1, y1, x2, y2): [" +
                   ", ".join(f"{v:.4f}" for v in box) + "]" for box in boxes]
        weights = [1.] + [0.] * (len(boxes) - 1)
        text = re.sub(r"Please provide the bounding box coordinate of the region", "Select the bounding box of the region", text, flags=re.I)
        extra["gold_bbox"] = boxes[0]
        protocol = "oracle_box_selection_synthetic_negatives"
    else:
        counts = answer_counts(sample["answer"])
        choices = pool.choices(counts, k, rng)
        weights = [counts.get(c, 0) / sum(counts.values()) for c in choices]
        extra["answer_votes"] = counts
        protocol = "gold_included_choice_train_pool"
    order = list(range(len(choices)))
    rng.shuffle(order)
    criteria = {f"option_{j}": choices[i] for j, i in enumerate(order)}
    target = {f"option_{j}": weights[i] for j, i in enumerate(order)}
    if task == "Grounding":
        extra["candidate_boxes"] = {f"option_{j}": boxes[i] for j, i in enumerate(order)}
    content = [{"type": "text", "text": "Use the provided information to make the requested decision."}]
    if sample["image"]:
        content.append({"type": "image_url", "image_url": {"url": sample["image"]}})
    record = {"group_id": sample["group"],
        "request": {"state": {"messages": [{"role": "user", "content": content}]},
                    "questions": {"q0": {"type": "choice", "instructions": text, "criteria": criteria}}},
        "targets": {"q0": {"probabilities": target}},
        "meta": {"record_id": sample["id"], "source_id": sample["source_id"], "source_row": sample.get("source_row"), "turn": sample["turn"],
                 "domain": task, "modality": "image" if sample["image"] else "text", "language_bucket": "en",
                 "source": "CoIN", "protocol": protocol, "split": split, "candidate_count": len(criteria), **extra}}
    return validate_record(record)


def prepare(source, image_root, output, tasks=TASKS, choices=8, seed=42, eval_source="official",
            holdout=0.1, annotations=None, allow_missing_images=False, limit=None):
    source, image_root, output = map(lambda p: Path(p).resolve(), (source, image_root, output))
    tasks = [ALIASES.get(t, t) for t in tasks]
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError("Tasks must be a nonempty unique subset of CoIN tasks")
    if type(choices) is not int or not 2 <= choices <= 255 or not 0 < holdout < 1:
        raise ValueError("Require 2–255 choices and 0 < holdout < 1")
    if eval_source not in {"official", "holdout"}:
        raise ValueError("eval_source must be official or holdout")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    manifest = dict(format="coin_decision_v1", seed=seed, choices=choices, tasks=[],
        eval_source=eval_source, holdout_fraction=holdout if eval_source == "holdout" else None,
        smoke_limit=limit, image_root=str(image_root), images_checked=not allow_missing_images,
        warning="Adapted fixed-choice benchmark; scores are not original CoIN scores.")
    training_groups, evaluation_groups, checked_images = set(), set(), set()
    try:
        for task in tasks:
            print(f"Processing {task} ...")
            train_path = source / task / "train.json"
            train = read_samples(train_path, task, image_root)
            source_files = {str(train_path): digest(train_path)}
            if eval_source == "holdout":
                # The same image is ALWAYS assigned the same split across tasks.
                is_test = lambda s: stable_seed(seed, s["group"]) / 2**64 < holdout
                test = [s for s in train if is_test(s)]
                train = [s for s in train if not is_test(s)]
            else:
                eval_path = source / task / ("val.json" if task in {"VQAv2", "TextVQA", "VizWiz"} else "test.json")
                annotation_path = (annotations or {}).get(task)
                labels = read_annotations(annotation_path) if annotation_path else {}
                if annotation_path:
                    source_files[str(Path(annotation_path).resolve())] = digest(annotation_path)
                test = read_samples(eval_path, task, image_root, labels)
                source_files[str(eval_path)] = digest(eval_path)
            if limit:
                train, test = train[:limit], test[:limit]
            if not train or not test:
                raise ValueError(f"{task}: empty train/evaluation split; use more groups or a different seed")
            pool = AnswerPool(train) if task not in {"ScienceQA", "Grounding"} else None
            folder = staging / task
            folder.mkdir()
            counts = {}
            for split, samples in (("train", train), ("test", test)):
                with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                    for sample in samples:
                        if sample["image"] and sample["image"] not in checked_images:
                            if not allow_missing_images and not Path(sample["image"]).is_file():
                                raise FileNotFoundError(sample["image"])
                            checked_images.add(sample["image"])
                        try:
                            record = convert(sample, task, pool, choices, seed, split)
                        except (ValueError, TypeError, KeyError) as exc:
                            raise ValueError(f"{task}/{split}/{sample['id']}: {exc}") from exc
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        (training_groups if split == "train" else evaluation_groups).add(record["group_id"])
                counts[split] = len(samples)
            manifest["tasks"].append(dict(name=task, train=f"{task}/train.jsonl", test=f"{task}/test.jsonl",
                counts=counts, source_sha256=source_files,
                train_sha256=digest(folder / "train.jsonl"), test_sha256=digest(folder / "test.jsonl")))
            if pool:
                (folder / "answer_pool.json").write_text(json.dumps(pool.all, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["cross_task_train_eval_overlapping_groups"] = len(training_groups & evaluation_groups)
        if eval_source == "holdout" and training_groups & evaluation_groups:
            raise AssertionError("Holdout group leakage")
        (staging / "sequence.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="CoIN Instructions_Original directory (OCRVQA spelling)")
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--choices", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-source", choices=("holdout", "official"), default="official")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--annotations", action="append", default=[], metavar="TASK=PATH",
                   help="Optional official VQA/TextVQA/VizWiz annotations or ID-to-answer mapping")
    p.add_argument("--allow-missing-images", action="store_true", help="Conversion only; images still required for training")
    p.add_argument("--limit", type=int, help="Explicit smoke subset per task/split")
    args = vars(p.parse_args())
    annotations = {}
    for entry in args.pop("annotations"):
        task, separator, path = entry.partition("=")
        task = ALIASES.get(task, task)
        if not separator or task not in TASKS or task in annotations:
            p.error("--annotations requires a unique TASK=PATH")
        annotations[task] = path
    manifest = prepare(**args, annotations=annotations)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
