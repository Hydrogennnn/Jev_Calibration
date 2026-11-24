"""Download authors' CL4VQA annotations; optionally download COCO 2014 images."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from .protocol import DATA_URL, FILE_IDS, TASKS, task_order


def valid_rows(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a nonempty JSON array: {path}")
    return len(data)


def download(output, tasks=TASKS):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install the download-only dependency: python -m pip install 'gdown>=5,<6'") from exc
    output = Path(output).resolve()
    report = {"source": DATA_URL, "files": {}}
    for split in ("train", "test"):
        for task in task_order(tasks):
            path = output / split / f"{split}_q_{task}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(".json.part")
                try:
                    result = gdown.download(id=FILE_IDS[split][task], output=str(temporary),
                                            quiet=False, use_cookies=False)
                    if result is None:
                        raise RuntimeError(f"Google Drive download failed; use the authors' link: {DATA_URL}")
                    valid_rows(temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            count = valid_rows(path)
            report["files"][str(path.relative_to(output))] = {
                "google_drive_id": FILE_IDS[split][task], "rows": count,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            print(f"{split}/{task}: {count} rows", flush=True)
    (output / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def download_images(root):
    """Extract each official archive only after curl has downloaded it completely."""
    root = Path(root).resolve()
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    for split in ("train2014", "val2014"):
        marker = root / f".{split}.complete"
        if marker.exists() and (root / split).is_dir():
            continue
        archive = archives / f"{split}.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".zip.part")
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--continue-at", "-",
                            "--output", str(temporary),
                            f"https://images.cocodataset.org/zips/{split}.zip"], check=True)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            bundle.extractall(root)
        marker.write_text("Downloaded from images.cocodataset.org and extracted successfully.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/CL4VQA")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--images", type=Path,
                        help="Also download ~19 GB of COCO archives and extract here; omit to reuse existing images")
    args = parser.parse_args()
    download(args.output, args.tasks)
    if args.images:
        download_images(args.images)


if __name__ == "__main__":
    main()
