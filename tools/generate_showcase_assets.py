"""Generate static figures used by the GitHub project showcase."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "HELLOWORLD"
ASSETS = ROOT / "docs" / "assets"


def parse_validation_report() -> list[dict[str, float | int | str]]:
    report_path = PROJECT / "results" / "test_report.txt"
    rows: list[dict[str, float | int | str]] = []
    in_table = False
    with report_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("Class") and "Precision" in stripped:
                in_table = True
                continue
            if not in_table or not stripped or stripped.startswith("-") or stripped.startswith("="):
                continue
            parts = stripped.split()
            if len(parts) != 5:
                continue
            class_name, precision, recall, f1, support = parts
            rows.append(
                {
                    "class_name": class_name,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": int(support),
                }
            )
    return rows


def save_training_curves() -> None:
    history_path = PROJECT / "logs" / "training_history_dual.json"
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)
    fig.suptitle("Dual-branch Training Dynamics", fontsize=15, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.28)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="s", label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.28)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(ASSETS / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def save_prediction_distribution() -> None:
    labels_path = PROJECT / "results" / "test_set_labels.csv"
    counts: Counter[str] = Counter()
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for video_id, class_name, class_id in csv.reader(handle):
            counts[class_name] += 1

    items = sorted(counts.items(), key=lambda item: item[1])
    class_names = [name for name, _ in items]
    values = [value for _, value in items]

    fig, axis = plt.subplots(figsize=(9.5, 9), dpi=180)
    colors = ["#4666ff" if value >= 20 else "#66a182" for value in values]
    axis.barh(class_names, values, color=colors)
    axis.set_xlabel("Number of test-set predictions")
    axis.set_title("Predicted Class Distribution on the Test Set", fontsize=14, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for y, value in enumerate(values):
        axis.text(value + 0.35, y, str(value), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "test_prediction_distribution.png", bbox_inches="tight")
    plt.close(fig)


def save_validation_error_focus() -> None:
    rows = parse_validation_report()
    imperfect = [row for row in rows if float(row["f1"]) < 0.9999]
    imperfect = sorted(imperfect, key=lambda row: float(row["f1"]))

    fig, axis = plt.subplots(figsize=(9.5, 3.8), dpi=180)
    names = [str(row["class_name"]) for row in imperfect]
    f1_scores = [float(row["f1"]) for row in imperfect]
    recall_misses = [
        round(int(row["support"]) * (1.0 - float(row["recall"]))) for row in imperfect
    ]

    bars = axis.barh(names, f1_scores, color=["#ef8354", "#ef8354", "#6a8dff", "#6a8dff"])
    axis.set_xlim(0.88, 1.005)
    axis.set_xlabel("Validation F1 score")
    axis.set_title("Validation Error Focus: Classes Below Perfect F1", fontsize=14, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for bar, row, misses in zip(bars, imperfect, recall_misses):
        label = f"P={float(row['precision']):.3f}, R={float(row['recall']):.3f}, missed={misses}"
        axis.text(
            min(float(row["f1"]) + 0.003, 0.998),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8.5,
        )

    fig.tight_layout()
    fig.savefig(ASSETS / "validation_error_focus.png", bbox_inches="tight")
    plt.close(fig)


def add_box(axis, xy, text, width=1.9, height=0.78, facecolor="#f7f8fb", edgecolor="#1f2937"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.5)
    return (x, y, width, height)


def add_arrow(axis, start, end):
    sx, sy = start
    ex, ey = end
    arrow = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.3,
        color="#374151",
    )
    axis.add_patch(arrow)


def save_pipeline_overview() -> None:
    fig, axis = plt.subplots(figsize=(13.2, 5.2), dpi=180)
    axis.set_xlim(0, 13.2)
    axis.set_ylim(0, 5)
    axis.axis("off")
    axis.set_title("Human-aware Action Recognition Pipeline", fontsize=16, fontweight="bold", pad=12)

    raw = add_box(axis, (0.45, 2.1), "Raw RGB\nvideos", facecolor="#eef2ff")
    sample = add_box(axis, (2.55, 2.1), "Uniform sampling\n16 frames", facecolor="#eff6ff")
    person = add_box(axis, (4.75, 3.15), "YOLOv8\nperson crops\n384 x 384", height=0.95, facecolor="#ecfdf5")
    arm = add_box(axis, (4.75, 0.9), "MediaPipe\narm crops\n192 x 192", height=0.95, facecolor="#fff7ed")
    r3d = add_box(axis, (7.15, 3.15), "R3D-18\nvideo logits", height=0.95, facecolor="#dbeafe")
    armcnn = add_box(axis, (7.15, 0.9), "SmallArmCNN\narm logits", height=0.95, facecolor="#ffedd5")
    fusion = add_box(axis, (9.45, 2.1), "Weighted fusion\n0.7 R3D + 0.3 Arm", width=2.0, facecolor="#f5f3ff")
    output = add_box(axis, (11.85, 2.1), "30-class\nprediction CSV", width=1.15, facecolor="#f8fafc")

    add_arrow(axis, (raw[0] + raw[2], raw[1] + raw[3] / 2), (sample[0], sample[1] + sample[3] / 2))
    add_arrow(axis, (sample[0] + sample[2], sample[1] + sample[3] / 2), (person[0], person[1] + person[3] / 2))
    add_arrow(axis, (sample[0] + sample[2], sample[1] + sample[3] / 2), (arm[0], arm[1] + arm[3] / 2))
    add_arrow(axis, (person[0] + person[2], person[1] + person[3] / 2), (r3d[0], r3d[1] + r3d[3] / 2))
    add_arrow(axis, (arm[0] + arm[2], arm[1] + arm[3] / 2), (armcnn[0], armcnn[1] + armcnn[3] / 2))
    add_arrow(axis, (r3d[0] + r3d[2], r3d[1] + r3d[3] / 2), (fusion[0], fusion[1] + fusion[3] / 2))
    add_arrow(axis, (armcnn[0] + armcnn[2], armcnn[1] + armcnn[3] / 2), (fusion[0], fusion[1] + fusion[3] / 2))
    add_arrow(axis, (fusion[0] + fusion[2], fusion[1] + fusion[3] / 2), (output[0], output[1] + output[3] / 2))

    axis.text(5.7, 4.35, "Global temporal evidence", ha="center", fontsize=9, color="#1f2937")
    axis.text(5.7, 0.35, "Local arm/tool evidence", ha="center", fontsize=9, color="#1f2937")
    fig.tight_layout()
    fig.savefig(ASSETS / "pipeline_overview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    save_training_curves()
    save_prediction_distribution()
    save_validation_error_focus()
    save_pipeline_overview()
    print(f"Showcase assets written to {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
