"""Run common project workflow stages from the repository root.

This script intentionally orchestrates existing entry points instead of replacing
them. It keeps the workflow visible and easy to audit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "HELLOWORLD"


STAGES: dict[str, list[str]] = {
    "preprocess-train": ["preprocess_videos.py"],
    "split": ["split_dataset.py"],
    "train": ["train_dual.py"],
    "validate": ["validate.py"],
    "preprocess-test": ["preprocess_test_video.py"],
    "predict-test": ["testing_set_application.py"],
}

PIPELINES: dict[str, list[str]] = {
    "all": [
        "preprocess-train",
        "split",
        "train",
        "validate",
        "preprocess-test",
        "predict-test",
    ],
    "submission": ["preprocess-test", "predict-test"],
    "check": [],
}


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"[pipeline] {cwd.relative_to(ROOT)} > {rendered}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def run_stage(stage: str, python_executable: str, dry_run: bool) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    run_command([python_executable, *STAGES[stage]], PROJECT_DIR, dry_run)


def run_checks(python_executable: str, dry_run: bool) -> None:
    run_command([python_executable, "tools/validate_repository.py"], ROOT, dry_run)
    source_files = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "HELLOWORLD").glob("*.py"))]
    source_files.extend(["run_pipeline.py", "tools/validate_repository.py", "tools/generate_showcase_assets.py"])
    run_command([python_executable, "-m", "py_compile", *source_files], ROOT, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HRI30 action-recognition workflow stages.")
    parser.add_argument(
        "target",
        choices=[*STAGES.keys(), *PIPELINES.keys()],
        help="Stage or pipeline to run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target == "check":
        run_checks(args.python, args.dry_run)
        return

    if args.target in PIPELINES:
        for stage in PIPELINES[args.target]:
            run_stage(stage, args.python, args.dry_run)
        return

    run_stage(args.target, args.python, args.dry_run)


if __name__ == "__main__":
    main()
