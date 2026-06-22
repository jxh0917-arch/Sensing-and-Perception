# Project Notes

## Short Explanation

This project addresses human activity recognition for collaborative robots in manufacturing. The pipeline uses a dual-branch design: an R3D-18 branch models whole-body spatiotemporal motion from sampled video frames, and a lightweight CNN models local arm crops extracted with MediaPipe. The two outputs are fused with a weighted sum, giving priority to global motion while preserving local manipulation cues.

## Contribution

The contribution is a task-informed fusion design rather than a single generic action classifier. It explicitly models two levels of evidence:

- global temporal dynamics for movement direction and action progression
- local arm/tool cues for manipulation-specific distinctions

This makes the model easier to justify and inspect than a single opaque end-to-end baseline.

## Evidence

- The workflow is reproducible from preprocessing to final CSV export.
- The validation report shows 99.05% accuracy and 0.9904 macro F1.
- Errors concentrate in directionally similar backward-motion classes, which is consistent with the visual ambiguity of the task.
- The repository separates source code, data placement, generated artifacts, and documentation.

## Likely Discussion Points

**Why not use only R3D-18?**  
R3D-18 is strong for global motion, but manufacturing actions can hinge on tool or hand posture. The arm branch adds a targeted local cue without replacing the temporal model.

**Why weighted late fusion?**  
Late fusion is simple, stable, and interpretable. It lets the dominant video branch carry temporal evidence while the arm branch acts as auxiliary evidence.

**What would improve the method next?**  
A stronger next step would add temporal arm modelling across all sampled frames, compare against single-branch baselines, and run ablations for fusion weights and crop strategies.

**What is the main limitation?**  
The local branch depends on pose visibility. Occlusion or poor camera angle can reduce arm-crop quality, so future work should include more robust keypoint tracking or learned attention over regions.
