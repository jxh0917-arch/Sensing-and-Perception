# Reproducibility Guide

This guide describes the complete workflow from raw videos to the final submission CSV.

## 1. Environment

The project was developed with:

- Python 3.12.10
- PyTorch 2.7.0 with CUDA 12.8
- torchvision 0.22.0
- NVIDIA RTX 3080

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Alternatively, create the Conda environment:

```bash
conda env create -f environment.yml
conda activate sensing-perception
```

If CUDA wheels are unavailable for your platform, install the matching PyTorch build from the official PyTorch selector, then reinstall the remaining packages in `HELLOWORLD/requirements.txt`.

See `docs/environment.md` for the upload policy: dependency specifications are committed, but the local Python runtime and virtual environments are not.

## 2. Data Placement

Place the supplied datasets in:

```text
HELLOWORLD/training_set/
HELLOWORLD/test_set/
```

The expected training labels are already stored in:

```text
HELLOWORLD/annotations/training_set_labels.csv
HELLOWORLD/annotations/classInd.txt
```

## 3. Preprocess Training Videos

```bash
cd HELLOWORLD
python preprocess_videos.py
```

This creates:

```text
HELLOWORLD/preprocessed_frames/
```

The output contains person crops for all sampled frames and arm crops for the first and last sampled frames.

## 4. Split the Dataset

```bash
python split_dataset.py
```

This writes:

```text
annotations/train.csv
annotations/val.csv
annotations/test.csv
```

## 5. Train the Dual Model

```bash
python train_dual.py
```

Checkpoints are written locally to:

```text
checkpoints/best_model.pth
checkpoints/last_model.pth
```

These files are intentionally ignored by Git because of size.

## 6. Validate

```bash
python validate.py
```

Outputs include:

```text
results/validation_report.txt
results/validation_confusion_matrix.png
results/validation_predictions.npz
```

Only lightweight text/CSV result summaries should be committed.

## 7. Generate Test Submission

Preprocess the hidden/test videos:

```bash
python preprocess_test_video.py
```

Run inference:

```bash
python testing_set_application.py
```

The required submission file is:

```text
results/test_set_labels.csv
```

It is saved without a header and follows:

```text
video_id,class_name,class_id
```

## 8. Repository Validation

From the repository root:

```bash
python tools/validate_repository.py
python -m py_compile HELLOWORLD/*.py
```

The GitHub Actions workflow runs these static checks automatically.

You can also use the repository-level workflow runner:

```bash
python run_pipeline.py check
python run_pipeline.py submission --dry-run
```

On Windows PowerShell:

```powershell
.\scripts\run_pipeline.ps1 check
.\scripts\run_pipeline.ps1 submission -DryRun
```
