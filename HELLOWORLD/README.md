# Technical Workflow: Dual-branch HRI30 Action Recognition

This directory contains the executable workflow for the human-aware action-recognition project.

For the academic overview, assessment alignment, and interview narrative, start from the repository-level `README.md` and the `docs/` directory.

## Core Idea

The model combines:

- `R3D-18`: global spatiotemporal action recognition from 16 person-centred frames.
- `SmallArmCNN`: local arm-crop recognition from MediaPipe-derived arm regions.
- `weighted_sum` fusion: `0.7 * R3D logits + 0.3 * ArmCNN logits`.

The fusion settings are centralised in `config.py`.

## Expected Local Data

Large files are intentionally not tracked by Git. Place them locally as follows:

```text
HELLOWORLD/
|-- training_set/                 # labelled training videos
|-- test_set/                     # unlabelled test videos
|-- checkpoints/                  # local model weights
|-- preprocessed_frames/          # generated training crops
`-- preprocessed_test_frames/     # generated test crops
```

## Commands

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Preprocess training videos:

```bash
python preprocess_videos.py
```

Create balanced train/validation/test CSV splits:

```bash
python split_dataset.py
```

Train:

```bash
python train_dual.py
```

Validate:

```bash
python validate.py
```

Preprocess unlabelled test videos and export predictions:

```bash
python preprocess_test_video.py
python testing_set_application.py
```

The coursework submission file is:

```text
results/test_set_labels.csv
```

## Important Files

- `config.py`: paths, hyperparameters, and fusion weights.
- `dual_model.py`: R3D-18, SmallArmCNN, fusion, and checkpoint loading.
- `preprocessed_dataset.py`: dataset loading and variable-length arm-image collation.
- `train_dual.py`: training loop, scheduler, checkpointing, and early stopping.
- `validate.py`: validation metrics and confusion-matrix export.
- `testing_set_application.py`: final test-set CSV generation.
- `results/test_report.txt`: current validation report.
- `results/test_set_labels.csv`: final test-set prediction output.
