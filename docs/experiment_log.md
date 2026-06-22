# Experiment Log

This log records implemented experiments and planned comparisons. It avoids reporting baseline scores that were not measured. A fuller comparison protocol is maintained in `docs/ablation_plan.md`.

## Completed Experiment

| ID | Model | Inputs | Fusion | Validation accuracy | Macro F1 | Notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| E1 | R3D-18 + SmallArmCNN | 16 person crops + first/last arm crops | Weighted late fusion, `0.7/0.3` | 99.05% | 0.9904 | Final submitted workflow |

## Configuration

| Setting | Value |
| --- | --- |
| Dataset split | class-balanced train/validation/test split |
| Frame sampling | 16 uniformly sampled frames |
| Person crop | YOLOv8n, `384 x 384` |
| Arm crop | MediaPipe Pose, `192 x 192` |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Scheduler | cosine annealing |
| Seed | `42` |

## Planned Ablation Protocol

The following comparisons would strengthen the evidence base if rerun time is available. See `docs/ablation_plan.md` for hypotheses and reporting fields.

| Planned ID | Model variant | Purpose |
| --- | --- | --- |
| A1 | R3D-18 only | Measure the contribution of global temporal features |
| A2 | SmallArmCNN only | Measure whether arm crops are informative alone |
| A3 | R3D-18 + ArmCNN, `0.8/0.2` | Test sensitivity to a smaller arm-branch weight |
| A4 | R3D-18 + ArmCNN, attention fusion | Compare fixed fusion with learned fusion |
| A5 | Arm crops from all sampled frames | Test whether richer local temporal evidence improves ambiguous classes |

## Interpretation

The completed experiment supports the main design claim: a global temporal model can be augmented with local arm/tool evidence while keeping the fusion rule interpretable. The planned ablations are included as a transparent next-step protocol rather than as unmeasured results.
