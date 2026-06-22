# Methodology

## Research Question

Can industrial human actions be recognised more reliably by combining whole-body temporal motion with local arm-pose cues?

The project assumes that many manufacturing actions are defined by both movement direction and tool interaction. A single video backbone may capture global motion, but tool use and hand-dominant actions often depend on smaller local regions. The proposed pipeline therefore separates the representation into two complementary streams.

## Preprocessing

Each video is uniformly sampled into 16 frames. For each sampled frame:

1. YOLOv8n detects person bounding boxes.
2. The person region is expanded into a centred `384 x 384` square crop.
3. MediaPipe Pose detects arm keypoints in the first and last sampled frames.
4. Arm regions are cropped into `192 x 192` images.
5. Missing person or arm evidence is handled with black placeholders so tensor shapes stay stable.

This creates a controlled input representation while preserving temporal order for the video branch.

## Model

The model has two branches:

- **R3D-18 branch**: a 3D CNN pretrained on action-recognition data, fine-tuned for the 30 HRI30 classes.
- **SmallArmCNN branch**: a lightweight 2D CNN that classifies arm crops and averages multiple arm logits per video.

The final class logits use weighted late fusion:

```text
final_logits = 0.7 * r3d_logits + 0.3 * arm_logits
```

The R3D branch receives the larger weight because it observes full temporal action structure. The arm branch is deliberately secondary: it provides local evidence without overwhelming the more informative global motion stream.

## Training

The training script uses:

- Adam optimizer
- learning rate `1e-4`
- weight decay `1e-4`
- cosine annealing learning-rate schedule
- cross-entropy loss
- gradient clipping at `1.0`
- early stopping patience of 5 epochs
- random seed `42`

The split script creates an 80/10/10 train/validation/test partition with class balance.

## Justification

The design is grounded in the structure of industrial activity recognition. Actions such as drilling, polishing, picking up tools, and carrying objects can differ through small limb and tool cues even when the coarse body trajectory is similar. Late fusion keeps the two evidence sources interpretable: the system can be explained as a global motion recogniser augmented by a local manipulation recogniser.

## Limitations

- The arm detector depends on MediaPipe keypoint visibility, so occlusion and camera angle can reduce the usefulness of the local branch.
- The current arm branch only uses first and last sampled frames, which may miss short mid-action manipulation cues.
- Validation accuracy is high, but hidden test performance should be treated as the stronger estimate of generalisation.
- Checkpoints are large and not stored in Git; a release asset or external model registry is recommended for long-term reproducibility.
