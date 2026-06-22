# Release Artifacts

Git should contain code, documentation, tracked result summaries, and small figures. Large datasets, raw videos, generated frame crops, local runtimes, and model checkpoints should remain outside the repository unless their redistribution rights are clear.

## Public Release Bundle

Create a lightweight public bundle with:

```bash
python tools/prepare_release_bundle.py
```

The script writes:

```text
dist/sensing_perception_public_artifacts.zip
dist/sensing_perception_public_artifacts.sha256
```

The bundle is intended for GitHub Releases. It contains the files needed to inspect the project workflow, reproduce the public checks, and review the submitted outputs.

## Included Categories

| Category | Examples |
| --- | --- |
| Project overview | `README.md`, `HELLOWORLD/README.md` |
| Documentation | `docs/*.md` |
| Figures | `docs/assets/*.png` |
| Source code | `HELLOWORLD/*.py`, `tools/*.py`, `run_pipeline.py` |
| Configuration | `requirements.txt`, `HELLOWORLD/requirements.txt`, `environment.yml` |
| Public labels and outputs | `HELLOWORLD/annotations/*.csv`, `HELLOWORLD/results/*.csv`, `HELLOWORLD/results/*.txt` |
| Training log | `HELLOWORLD/logs/training_history_dual.json` |

## Excluded Categories

| Category | Reason |
| --- | --- |
| Raw videos | Large files and possible redistribution restrictions |
| Training/test archives | Too large for normal Git history and not needed for code review |
| Preprocessed frames | Generated artifacts reproducible from raw videos |
| Checkpoints | Better attached separately after rights and size constraints are confirmed |
| Local Python runtime | Replaced by dependency specifications |

## Recommended GitHub Release Text

```text
Title: Public project artifact bundle

This release contains the public documentation, source code, result summaries, figures, labels, and workflow scripts for the HRI30 action-recognition project.

It intentionally excludes raw videos, generated frame crops, local runtimes, and model checkpoints. Dataset access and large artifacts should be handled through the approved course or institutional channel.

Validation:
- python run_pipeline.py check
- GitHub Actions: Repository checks
```

## Checksum

The `.sha256` file lets readers verify that the downloaded bundle has not changed. If a model checkpoint is released separately later, add a separate checksum for that file as well.
