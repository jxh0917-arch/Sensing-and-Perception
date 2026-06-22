# Literature Context

This project combines ideas from video action recognition, pose-assisted perception, and industrial human-robot interaction.

## Video Action Recognition

The R3D-18 branch is motivated by work on spatiotemporal convolutional networks for video. Tran et al. study 3D and factorised spatiotemporal convolutions for action recognition and show why temporal convolution is useful beyond frame-wise 2D CNNs: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248).

Large-scale pretraining is important for video recognition. The Kinetics dataset introduced broad human action video supervision with 400 action classes: [The Kinetics Human Action Video Dataset](https://arxiv.org/abs/1705.06950). Carreira and Zisserman further discuss Kinetics pretraining and I3D-style video architectures in [Quo Vadis, Action Recognition?](https://arxiv.org/abs/1705.07750).

## Pose and Local Motion Cues

The arm branch is motivated by the observation that manufacturing actions can depend on small local cues around hands, arms, and tools. MediaPipe/BlazePose provides efficient body pose estimation suitable for practical pipelines. BlazePose is described in [BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/abs/2006.10204), and the later holistic model is described in [BlazePose GHUM Holistic](https://arxiv.org/abs/2206.11678).

In this project, pose is not used as the final representation. Instead, pose keypoints guide local arm cropping, and a CNN learns visual evidence from the cropped arm regions.

## Industrial Human-Robot Interaction

Industrial HRI action recognition has different constraints from generic internet video recognition. Actions can be visually similar, tool-specific, and strongly shaped by collaboration context. Recent work such as [CEFHRI](https://arxiv.org/abs/2308.14965) highlights HRI30 as one of the benchmark datasets for industrial human-robot interaction recognition and discusses data privacy and communication constraints in industrial settings.

## Position of This Project

This repository implements a practical late-fusion pipeline:

- R3D-18 captures global temporal motion.
- YOLOv8n isolates person-centred visual evidence before sequence modelling.
- MediaPipe-guided arm crops provide local manipulation cues.
- Weighted fusion keeps the model interpretable and easy to ablate.

The design is intentionally incremental: it extends a strong video backbone with a task-specific local cue branch, which is appropriate for manufacturing actions where tool and arm context matter.
