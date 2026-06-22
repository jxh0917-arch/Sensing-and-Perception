"""Create a public release bundle without raw datasets or checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
DEFAULT_ZIP = DIST_DIR / "sensing_perception_public_artifacts.zip"


EXPLICIT_FILES = [
    ".github/workflows/ci.yml",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "environment.yml",
    "requirements.txt",
    "run_pipeline.py",
    "scripts/run_pipeline.ps1",
    "HELLOWORLD/README.md",
    "HELLOWORLD/requirements.txt",
    "HELLOWORLD/logs/training_history_dual.json",
    "HELLOWORLD/results/test_report.txt",
    "HELLOWORLD/results/test_set_labels.csv",
    "HELLOWORLD/results/test_set_labels_detailed.csv",
]

GLOB_PATTERNS = [
    "HELLOWORLD/*.py",
    "HELLOWORLD/annotations/*.csv",
    "HELLOWORLD/annotations/*.txt",
    "docs/*.md",
    "docs/assets/*.png",
    "tools/*.py",
]

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "dist",
    "python",
    "training_set",
    "test_set",
    "preprocessed_frames",
    "preprocessed_test_frames",
    "checkpoints",
}

EXCLUDED_SUFFIXES = {
    ".avi",
    ".mp4",
    ".mov",
    ".rar",
    ".zip",
    ".7z",
    ".pt",
    ".pth",
    ".pyc",
}


def should_exclude(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def collect_files() -> list[Path]:
    files: set[Path] = set()

    for item in EXPLICIT_FILES:
        path = ROOT / item
        if not path.exists():
            raise FileNotFoundError(f"Required release file is missing: {item}")
        files.add(path)

    for pattern in GLOB_PATTERNS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                files.add(path)

    included = sorted(path for path in files if not should_exclude(path))
    if not included:
        raise RuntimeError("No files were selected for the release bundle")
    return included


def build_manifest(files: list[Path]) -> str:
    lines = [
        "Sensing and Perception public artifact bundle",
        "",
        "This bundle intentionally excludes raw videos, generated frame crops, checkpoints, archives, and local runtimes.",
        "",
        "Included files:",
    ]
    lines.extend(f"- {path.relative_to(ROOT).as_posix()}" for path in files)
    lines.append("")
    return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(output: Path) -> Path:
    files = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr("RELEASE_MANIFEST.txt", build_manifest(files))

    checksum = sha256(output)
    checksum_path = output.with_suffix(".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")

    print(f"Created {output.relative_to(ROOT)}")
    print(f"Created {checksum_path.relative_to(ROOT)}")
    print(f"Included {len(files)} public files")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a public GitHub Release artifact bundle.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ZIP,
        help="Output zip path. Defaults to dist/sensing_perception_public_artifacts.zip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    create_bundle(output)


if __name__ == "__main__":
    main()
