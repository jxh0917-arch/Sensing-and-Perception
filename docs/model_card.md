# Model Card

## Model Summary

The system is a dual-branch action-recognition model for industrial human-robot interaction videos. It combines an R3D-18 video branch with a lightweight arm-crop CNN and produces one of 30 HRI30 action classes.

## Intended Use

The model is intended for recognising human activities in manufacturing-style video recordings, especially actions involving movement direction, tool use, and object handling.

Appropriate uses:

- coursework evaluation on the supplied 7CCEMSAP/HRI30-style dataset
- research demonstrations of human-aware perception for collaborative robots
- reproducibility studies of dual-stream action-recognition pipelines

Out-of-scope uses:

- safety-critical robot control without additional validation
- deployment on unseen camera setups without domain testing
- identity recognition or person profiling

## Inputs and Outputs

Input:

- RGB videos from the manufacturing action-recognition dataset
- 16 uniformly sampled frames per video
- person crops of `384 x 384`
- arm crops of `192 x 192` from MediaPipe Pose detections

Output:

- class logits for 30 action classes
- final CSV predictions in `video_id,class_name,class_id` format

## Architecture

- **R3D-18 branch**: extracts whole-body spatiotemporal features.
- **SmallArmCNN branch**: extracts local arm/tool cues.
- **Weighted late fusion**: combines branch logits with `0.7 * R3D + 0.3 * ArmCNN`.

## Training Configuration

- Optimizer: Adam
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Scheduler: cosine annealing
- Loss: cross-entropy
- Gradient clipping: `1.0`
- Random seed: `42`
- Batch size: `2`
- Epochs recorded: `9`

## Evaluation Snapshot

Internal validation report:

- Overall accuracy: `99.05%`
- Macro precision: `0.9917`
- Macro recall: `0.9905`
- Macro F1: `0.9904`
- Micro F1: `0.9905`

The reported validation split contains 7 examples per class across 30 classes.

## Known Limitations

- The arm branch depends on pose visibility and can degrade under occlusion.
- Arm evidence is extracted from the first and last sampled frames only.
- The validation score should not be treated as a deployment guarantee.
- Raw training and test videos are not included in the public repository.
- The trained checkpoint is not tracked in Git because of size.

## Reproducibility

Reproduction instructions are in `docs/reproducibility.md`. Environment files are provided through `requirements.txt`, `HELLOWORLD/requirements.txt`, and `environment.yml`.

Dataset scope and public/private boundaries are documented in `docs/data_card.md`.
