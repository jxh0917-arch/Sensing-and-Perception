# Artifact Manifest

This manifest explains which project artifacts are public in GitHub and which artifacts remain local because of size, reproducibility, or data-governance reasons.

## Public GitHub Artifacts

| Artifact | Path | Purpose |
| --- | --- | --- |
| Main project narrative | `README.md` | Academic overview and quick start |
| Technical workflow | `HELLOWORLD/README.md` | Execution commands and file roles |
| Methodology | `docs/methodology.md` | Model rationale and limitations |
| Reproducibility guide | `docs/reproducibility.md` | End-to-end rerun instructions |
| Results summary | `docs/results.md` | Validation metrics and interpretation |
| Interview notes | `docs/phd_interview_notes.md` | Compact PhD interview narrative |
| Showcase page | `docs/showcase.md` | Visual project walkthrough |
| Environment guide | `docs/environment.md` | Reproducible setup and upload policy |
| Conda environment | `environment.yml` | Optional environment recreation file |
| Prediction CSV | `HELLOWORLD/results/test_set_labels.csv` | Required final output format |
| Detailed predictions | `HELLOWORLD/results/test_set_labels_detailed.csv` | Confidence analysis |
| Validation report | `HELLOWORLD/results/test_report.txt` | Per-class metrics |

## Local-only Artifacts

| Artifact | Approximate size | Reason |
| --- | ---: | --- |
| `HELLOWORLD/training_set.rar` | 9.4 GB | Too large for GitHub Git/LFS limits on a free account |
| `Individual.rar` | 907 MB | Large archive, not needed for code review |
| `HELLOWORLD/checkpoints/best_model.pth` | 399 MB | Better distributed as a release/model artifact after rights are confirmed |
| `HELLOWORLD/test_set/*.avi` | 6.8 GB total | Raw video data should not be placed in a public repository |
| `HELLOWORLD/preprocessed_test_frames/` | 1.5 GB total | Generated data, reproducible from raw videos |
| `HELLOWORLD/python/` | many GB | Local runtime and dependency cache; replaced by `requirements.txt` |

## Recommended Sharing Model

For a public academic repository, keep code, documentation, result summaries, and generated figures in Git. Store large or restricted artifacts separately:

- Use GitHub Releases only for redistributable model checkpoints or small demonstration bundles.
- Use institutional storage, OneDrive, Google Drive, Zenodo, or a course-provided data link for raw datasets.
- Add checksums and download instructions if an artifact is necessary for exact reproduction.

This preserves reproducibility without making the repository slow, fragile, or legally ambiguous.
