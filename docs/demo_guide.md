# Demo Guide

This guide describes how to present the repository as a compact technical demonstration without publishing restricted raw videos.

## Recommended Walkthrough

| Step | What to show | Why it matters |
| --- | --- | --- |
| 1 | `README.md` project summary and badges | Establishes the task, method, and repository health. |
| 2 | Pipeline figure in the README showcase | Explains the dual-branch design visually. |
| 3 | `docs/data_card.md` | Shows the public/private data boundary and responsible dataset handling. |
| 4 | `docs/literature_context.md` | Connects the implementation to video action recognition and pose-assisted perception. |
| 5 | `docs/results.md` and `docs/error_analysis.md` | Moves from headline metrics to class-level interpretation. |
| 6 | `docs/ablation_plan.md` | Shows how the method could be tested further. |
| 7 | `python run_pipeline.py check` | Demonstrates that the repository has a verifiable workflow. |

## Live Commands

From the repository root:

```bash
python run_pipeline.py check
python run_pipeline.py submission --dry-run
python tools/prepare_release_bundle.py
```

The first command validates the repository structure and compiles Python sources. The second command shows the final submission path without running heavy preprocessing. The third command creates a release-ready public artifact bundle under `dist/`.

## Optional Visual Demo

If rights-cleared sample clips are available, a short demo can be prepared with three panels:

| Panel | Content |
| --- | --- |
| Input | A short RGB clip or a few sampled frames |
| Perception | Person-centred crop and arm crop examples |
| Prediction | Predicted class, confidence, and final CSV row |

Only use clips that are explicitly approved for public display. If raw video redistribution is not permitted, use the existing static figures in `docs/assets/` instead.

## Suggested Narrative

The strongest presentation angle is:

1. Industrial actions require both global motion and local manipulation cues.
2. R3D-18 models full-body temporal structure.
3. MediaPipe-guided arm crops add local tool/arm evidence.
4. Weighted late fusion keeps the system interpretable.
5. Error analysis identifies the next research step: direction-sensitive motion cues.

This keeps the demonstration focused on method, evidence, and reproducibility.
