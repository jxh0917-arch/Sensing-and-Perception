# Artifact Manifest

This manifest explains which project artifacts are public in GitHub and which artifacts remain local because of size, reproducibility, or data-governance reasons.

## Public GitHub Artifacts

| Artifact | Path | Purpose |
| --- | --- | --- |
| Main project narrative | `README.md` | Academic overview and quick start |
| Technical workflow | `HELLOWORLD/README.md` | Execution commands and file roles |
| Methodology | `docs/methodology.md` | Model rationale and limitations |
| Data card | `docs/data_card.md` | Dataset scope and public/private boundary |
| Literature context | `docs/literature_context.md` | Related work and method context |
| Ablation plan | `docs/ablation_plan.md` | Measured result and planned comparisons |
| Demo guide | `docs/demo_guide.md` | Walkthrough and public demo boundaries |
| Release artifact guide | `docs/release_artifacts.md` | Public bundle contents and release policy |
| Reproducibility guide | `docs/reproducibility.md` | End-to-end rerun instructions |
| Results summary | `docs/results.md` | Validation metrics and interpretation |
| Model card | `docs/model_card.md` | Intended use, inputs, outputs, and limitations |
| Experiment log | `docs/experiment_log.md` | Completed experiment and planned ablation protocol |
| Error analysis | `docs/error_analysis.md` | Validation error focus |
| Project notes | `docs/project_notes.md` | Compact project discussion notes |
| README showcase | `README.md` | Visual project walkthrough |
| Environment guide | `docs/environment.md` | Reproducible setup and upload policy |
| Conda environment | `environment.yml` | Optional environment recreation file |
| Workflow runner | `run_pipeline.py` | Runs common workflow stages from the repository root |
| Release bundler | `tools/prepare_release_bundle.py` | Creates a release-ready zip without large private artifacts |
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
- Use `python tools/prepare_release_bundle.py` to package the public project artifacts for a release.
- Use institutional storage, OneDrive, Google Drive, Zenodo, or a course-provided data link for raw datasets.
- Add checksums and download instructions if an artifact is necessary for exact reproduction.

This preserves reproducibility without making the repository slow, fragile, or legally ambiguous.
