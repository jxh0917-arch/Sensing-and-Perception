# Environment and Upload Policy

This project should upload reproducibility specifications, not the local runtime itself.

## Upload These

| Item | Path | Why it is useful |
| --- | --- | --- |
| Python dependency list | `HELLOWORLD/requirements.txt` | Direct `pip` install path |
| Root dependency pointer | `requirements.txt` | Convenience install from repository root |
| Conda environment file | `environment.yml` | Recreates a named environment |
| Experiment configuration | `HELLOWORLD/config.py` | Hyperparameters, paths, model settings |
| GitHub Actions checks | `.github/workflows/ci.yml` | Verifies repository structure and Python syntax |
| Reproducibility guide | `docs/reproducibility.md` | Full workflow from data placement to CSV export |

These files are lightweight, reviewable, and sufficient for another researcher to recreate the software environment.

## Do Not Upload These

| Item | Example path | Reason |
| --- | --- | --- |
| Local Python runtime | `HELLOWORLD/python/` | Huge, machine-specific, and reproducible from dependency files |
| Virtual environments | `.venv/`, `venv/` | Machine-specific generated state |
| Package caches | `__pycache__/`, `.pytest_cache/`, `.mplconfig/` | Generated files |
| Raw datasets | `training_set/`, `test_set/` | Large and potentially restricted coursework data |
| Preprocessed frames | `preprocessed_frames/`, `preprocessed_test_frames/` | Generated data, reproducible from raw videos |
| Model checkpoints | `checkpoints/*.pth` | Large binary artifacts; better handled through releases or external storage |

## Recommended Setup

Using `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate sensing-perception
```

If CUDA 12.8 wheels are not available on the target machine, install the matching PyTorch build first, then install the remaining packages from `HELLOWORLD/requirements.txt`.

## Reproducibility Boundary

The public GitHub repository contains all code, configuration, documentation, and result summaries. The private/local boundary contains raw videos, generated crops, checkpoints, and local runtime files. This is the right split for a public academic repository because it keeps the work inspectable while avoiding oversized and potentially restricted artifacts.
