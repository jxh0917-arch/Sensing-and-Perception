# Data Card

## Dataset Context

The project uses the HRI30-style industrial human-robot interaction action-recognition dataset supplied for the 7CCEMSAP Sensing and Perception final project. The public repository includes labels, class definitions, split files, result summaries, and code, but it does not include the raw videos.

## Task

The task is 30-class human activity recognition from RGB video. Each sample is a short video of a person performing a manufacturing-relevant action such as moving with a tool, picking up or putting down an object, drilling, polishing, or walking.

## Public Label Files

| File | Purpose |
| --- | --- |
| `HELLOWORLD/annotations/classInd.txt` | Mapping from class IDs to class names |
| `HELLOWORLD/annotations/training_set_labels.csv` | Original labelled training metadata |
| `HELLOWORLD/annotations/train.csv` | Class-balanced training split |
| `HELLOWORLD/annotations/val.csv` | Class-balanced validation split |
| `HELLOWORLD/annotations/test.csv` | Internal class-balanced test split |
| `HELLOWORLD/results/test_set_labels.csv` | Final prediction file in the required submission format |

CSV rows follow:

```text
video_id,class_name,class_id
```

## Local-only Data

Raw videos, preprocessed frame crops, and compressed archives remain local. They are excluded because they are large, generated from source data, and may be governed by coursework distribution rules.

Expected local placement:

```text
HELLOWORLD/training_set/
HELLOWORLD/test_set/
HELLOWORLD/preprocessed_frames/
HELLOWORLD/preprocessed_test_frames/
```

## Preprocessing

The pipeline converts raw videos into model-ready inputs:

- uniformly sample 16 frames per video
- detect person regions with YOLOv8n
- crop person-centred `384 x 384` images
- detect arm keypoints with MediaPipe Pose
- crop `192 x 192` arm images from first and last sampled frames
- store generated crops locally

## Ethical and Practical Notes

- The videos contain humans, so the repository does not redistribute raw visual data.
- The public files are sufficient to inspect the method and output format, but exact retraining requires access to the supplied dataset.
- Reported validation metrics apply to the provided development split and should not be treated as deployment evidence.
