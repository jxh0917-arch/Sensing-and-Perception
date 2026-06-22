# Project Showcase

This page is designed for quick review during a PhD interview or project discussion. It collects the complete public-facing evidence for the project without uploading the raw coursework dataset or local Python runtime.

## Pipeline

![Human-aware action recognition pipeline](assets/pipeline_overview.png)

The system combines a global temporal branch and a local arm branch. R3D-18 models whole-body motion from 16 sampled person crops, while SmallArmCNN models arm/tool cues from MediaPipe-derived crops. The final prediction uses weighted late fusion.

## Training Evidence

![Training and validation curves](assets/training_curves.png)

The training history shows fast convergence and a strong validation trajectory across the recorded epochs. The best validation report records 99.05% overall accuracy and 0.9904 macro F1.

## Test-set Prediction Distribution

![Test-set prediction distribution](assets/test_prediction_distribution.png)

The exported test prediction file is:

```text
HELLOWORLD/results/test_set_labels.csv
```

It follows the required coursework format:

```text
video_id,class_name,class_id
```

## What Is Included Publicly

- Full source code for preprocessing, training, validation, and inference.
- Class mapping and train/validation/test CSV split files.
- Training-history JSON and validation report.
- Final test-set prediction CSV and detailed probability CSV.
- Reproducibility, methodology, results, and interview notes.
- GitHub Actions workflow for repository-level checks.

## What Remains Local

The following are intentionally not committed to the public Git repository:

- raw training videos
- raw test videos
- preprocessed frame crops
- model checkpoints
- compressed dataset archives
- bundled local Python runtime

This keeps the repository reviewable, cloneable, and compliant with GitHub's large-file constraints while still showing the complete scientific workflow.
