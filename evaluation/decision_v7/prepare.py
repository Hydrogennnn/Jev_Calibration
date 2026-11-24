"""Convert decision-v7 to Valen JSONL and a source-based continual sequence.

Uses only the standard library and Valen's schema; no model or torch is needed.
Original train/development/calibration/test membership and multi-question states
are retained. The sequence's test pointer selects the requested evaluation split.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from valen.data.schema import candidates, validate_record


BENCHMARK = "decision_v7_source_cl_v1"
SPLITS = ("train", "development", "calibration", "test")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid source/task directory name: {value!r}")
    return value


def description(value, key):
    """Null means a named category, not an absent candidate."""
    if value is None:
        return key
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"Empty candidate description for {key}")
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def convert(row, split, line_number):
    original_meta = row["_meta"]
    source = source_name(original_meta["source"])
    identifier, group = original_meta["id"], original_meta["group_id"]
    if not all(isinstance(value, str) and value.strip() for value in (identifier, group)):
        raise ValueError("_meta.id and _meta.group_id must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)):
        raise ValueError("Expected a text, object or array state")
    # decision-v7 objects/arrays are structured text, not Valen messages objects.
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
    questions, targets, question_sources = {}, {}, {}
    if not isinstance(row["questions"], dict) or not row["questions"]:
        raise ValueError("questions must be a nonempty mapping")
    for qid, raw in row["questions"].items():
        if not isinstance(qid, str) or not qid:
            raise ValueError("Question IDs must be nonempty strings")
        kind, label = raw["type"], raw["label"]
        instructions = raw["instructions"]
        if isinstance(instructions, (dict, list)) and instructions:
            instructions = json.dumps(instructions, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"{qid}: instructions must be nonempty text or a structured instruction")
        question = {"type": kind, "instructions": instructions}
        criteria = raw.get("criteria")
        if kind == "noul":
            if type(label) is not bool:
                raise ValueError(f"{qid}: Noul label must be boolean")
            winner = "true" if label else "false"
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                    raise ValueError(f"{qid}: Noul criteria must define true and false")
                # Noul candidate descriptions are fixed in Valen; retain the
                # source-specific rubric in instructions, as in JevBench.
                question["instructions"] += (
                    "\nTrue / yes criterion: " + description(criteria["true"], "true")
                    + "\nFalse / no criterion: " + description(criteria["false"], "false"))
        elif kind == "choice":
            if not isinstance(criteria, dict) or not isinstance(label, str) or label not in criteria:
                raise ValueError(f"{qid}: Choice label must be a candidate key")
            question["criteria"] = {key: description(value, key) for key, value in criteria.items()}
            winner = label
        elif kind == "score":
            if not isinstance(criteria, list) or type(label) is not int or not 0 <= label < len(criteria):
                raise ValueError(f"{qid}: Score label must be a zero-based criteria index")
            question["criteria"] = [description(value, str(i)) for i, value in enumerate(criteria)]
            winner = str(label)
        else:
            raise ValueError(f"{qid}: Unsupported question type: {kind}")
        keys = [key for key, _ in candidates(question)]
        questions[qid] = question
        targets[qid] = {"probabilities": {key: float(key == winner) for key in keys}}
        question_sources[qid] = raw.get("src")
    return validate_record({"group_id": group,
        "request": {"state": state, "questions": questions}, "targets": targets,
        "meta": {"record_id": f"decision-v7:{source}:{split}:{identifier}",
                 "source_id": identifier, "source": source, "domain": source,
                 "split": split, "source_line": line_number, "modality": "text",
                 "language_bucket": original_meta.get("language_bucket", "unknown"),
                 "protocol": BENCHMARK, "question_sources": question_sources,
                 "original_meta": deepcopy(original_meta)}})


def prepare(source, output, tasks=None, eval_split="development"):
    if eval_split not in {"development", "test"}:
        raise ValueError("eval_split must be development or test")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if source in output.parents:
        raise ValueError("Output must be outside the input dataset directory")
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_files = {}
    for split in SPLITS:
        path = source / f"{split}.jsonl"
        sha = digest(path)
        expected = source_manifest.get("files", {}).get(path.name, {}).get("sha256")
        if expected is not None and sha != expected:
            raise ValueError(f"Input checksum mismatch: {path}")
        source_files[path.name] = {"sha256": sha}

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    counts = defaultdict(Counter)
    question_counts = defaultdict(Counter)
    kinds = defaultdict(lambda: defaultdict(Counter))
    groups = {split: set() for split in SPLITS}
    try:
        with ExitStack() as stack:
            streams = {}
            for split in SPLITS:
                path = source / f"{split}.jsonl"
                seen_ids = set()
                with path.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            record = convert(json.loads(line), split, line_number)
                            name = record["meta"]["source"]
                            identifier = record["meta"]["record_id"]
                            if identifier in seen_ids:
                                raise ValueError(f"Duplicate record ID: {identifier}")
                            seen_ids.add(identifier)
                            if (name, split) not in streams:
                                folder = staging / name
                                folder.mkdir(exist_ok=True)
                                streams[name, split] = stack.enter_context(
                                    (folder / f"{split}.jsonl").open("w", encoding="utf-8"))
                            streams[name, split].write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                            counts[name][split] += 1
                            question_counts[name][split] += len(record["request"]["questions"])
                            kinds[name][split].update(q["type"] for q in record["request"]["questions"].values())
                            groups[split].add(record["group_id"])
                        except (ValueError, KeyError, TypeError, AttributeError) as exc:
                            raise ValueError(f"{path}:{line_number}: {exc}") from exc
                source_files[path.name].update(records=len(seen_ids),
                    questions=sum(value[split] for value in question_counts.values()))
                for key in ("records", "questions"):
                    expected = source_manifest.get("files", {}).get(path.name, {}).get(key)
                    if expected is not None and source_files[path.name][key] != expected:
                        raise ValueError(f"Input {key} count mismatch: {path}")
        # Check every original split/source, including sources outside the chosen
        # sequence. Never move sibling/variant groups between train and evaluation.
        overlap = {}
        for i, first in enumerate(SPLITS):
            for second in SPLITS[i + 1:]:
                count = len(groups[first] & groups[second])
                overlap[f"{first}:{second}"] = count
                if count:
                    raise ValueError(f"Split group overlap: {first}/{second}: {count}")
        trainable = {name for name, values in counts.items() if values["train"]}
        if tasks is None:
            declared = source_manifest.get("trainable_sources", [])
            tasks = list(dict.fromkeys(name for name in declared if name in trainable))
            tasks += sorted(trainable - set(tasks))
        else:
            tasks = list(tasks)
        if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - trainable:
            raise ValueError(f"Tasks must be unique sources with training records: {sorted(trainable)}")
        missing = [name for name in tasks if not counts[name][eval_split]]
        if missing:
            raise ValueError(f"No {eval_split} records for {missing}; select --eval-split development "
                             "or explicitly choose --tasks with this evaluation split")
        # Preserve all sources and splits even when only a subset is trained.
        files = {}
        for name in sorted(counts):
            files[name] = {split: {"path": f"{name}/{split}.jsonl",
                "sha256": digest(staging / name / f"{split}.jsonl"),
                "records": counts[name][split], "questions": question_counts[name][split],
                "primitive_counts": dict(kinds[name][split])}
                for split in SPLITS if counts[name][split]}
        sequence = []
        for name in tasks:
            train, evaluation = files[name]["train"], files[name][eval_split]
            sequence.append({"name": name, "source": name, "abbreviation": name,
                "train": train["path"], "test": evaluation["path"],
                "train_sha256": train["sha256"], "test_sha256": evaluation["sha256"],
                "eval_split": eval_split,
                "counts": {"train": train["records"], "test": evaluation["records"]},
                "question_counts": {"train": train["questions"], "test": evaluation["questions"]}})
        manifest = {"format": "coin_decision_v1", "benchmark": BENCHMARK,
            "source": str(source), "source_manifest_sha256": digest(source_manifest_path),
            "source_files": source_files, "eval_source": eval_split, "tasks": sequence,
            "eval_only_sources": sorted(set(counts) - trainable),
            "unselected_train_sources": sorted(trainable - set(tasks)), "files": files,
            "split_group_overlaps": overlap, "cross_task_train_eval_overlapping_groups": 0,
            "warning": "Source-based adaptation. All original splits are retained; sequence test paths "
                       "refer to eval_source. Candidate descriptions are normalized, labels stay outside "
                       "requests. Group checks do not certify semantic or pretraining decontamination."}
        write_json(staging / "source_manifest.json", source_manifest)
        write_json(staging / "sequence.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/v7/decision-v7")
    parser.add_argument("--output", default="data/decision-v7-cl")
    parser.add_argument("--tasks", nargs="+", help="Source names in training order; default: input manifest order")
    parser.add_argument("--eval-split", choices=("development", "test"), default="development",
                        help="Evaluation split for every task; never falls back or resplits training data")
    manifest = prepare(**vars(parser.parse_args()))
    print(json.dumps({"benchmark": BENCHMARK, "eval_source": manifest["eval_source"],
        "tasks": [{"name": t["name"], **t["counts"]} for t in manifest["tasks"]],
        "eval_only_sources": manifest["eval_only_sources"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
