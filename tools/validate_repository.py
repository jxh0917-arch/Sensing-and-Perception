"""Lightweight repository checks for the HRI30 action-recognition project."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "HELLOWORLD"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path is missing: {path.relative_to(ROOT)}")


def read_class_ids(path: Path) -> set[int]:
    class_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            class_id, _ = stripped.split(" ", 1)
            class_ids.add(int(class_id))
    return class_ids


def validate_label_csv(path: Path, class_ids: set[int]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"{path.relative_to(ROOT)} is empty")

    for row_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            raise ValueError(
                f"{path.relative_to(ROOT)} row {row_number} has {len(row)} columns, expected 3"
            )
        if row[0].lower() == "video_id":
            raise ValueError(f"{path.relative_to(ROOT)} must not contain a header row")
        try:
            class_id = int(row[2])
        except ValueError as exc:
            raise ValueError(
                f"{path.relative_to(ROOT)} row {row_number} has a non-integer class id"
            ) from exc
        if class_id not in class_ids:
            raise ValueError(
                f"{path.relative_to(ROOT)} row {row_number} uses unknown class id {class_id}"
            )


def main() -> None:
    required_files = [
        PROJECT / "config.py",
        PROJECT / "dual_model.py",
        PROJECT / "preprocessed_dataset.py",
        PROJECT / "preprocess_videos.py",
        PROJECT / "preprocess_test_video.py",
        PROJECT / "train_dual.py",
        PROJECT / "validate.py",
        PROJECT / "testing_set_application.py",
        PROJECT / "annotations" / "classInd.txt",
        PROJECT / "annotations" / "training_set_labels.csv",
        PROJECT / "results" / "test_set_labels.csv",
        PROJECT / "results" / "test_report.txt",
        ROOT / "docs" / "reproducibility.md",
        ROOT / "docs" / "methodology.md",
    ]
    for path in required_files:
        require(path)

    class_ids = read_class_ids(PROJECT / "annotations" / "classInd.txt")
    if len(class_ids) != 30:
        raise ValueError(f"Expected 30 classes, found {len(class_ids)}")

    validate_label_csv(PROJECT / "annotations" / "training_set_labels.csv", class_ids)
    validate_label_csv(PROJECT / "results" / "test_set_labels.csv", class_ids)

    print("Repository checks passed.")


if __name__ == "__main__":
    main()
