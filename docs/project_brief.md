# Project Brief

## Context

The 7CCEMSAP final project asks teams to build a perception algorithm for recognising human activities in manufacturing videos. The motivation is human-aware collaborative robotics: robots need to infer human actions from visual signals so they can adapt proactively in shared industrial workspaces.

## Assessment Alignment

The coursework assessment gives most weight to reproducibility:

- **Reproducibility, 50%**: clear code, transparent workflow, complete documentation, and repeatable outputs.
- **Originality, 25%**: a meaningful idea beyond a basic off-the-shelf classifier.
- **Justification, 25%**: evidence-based design choices, awareness of relevant literature, and defensible parameters.

This repository is structured around those criteria. Code and workflow files support reproducibility, the dual-branch architecture supports originality, and the documentation explains the methodological choices.

## Deliverables

The expected technical deliverables are:

- source code for preprocessing, training, validation, and test inference
- final test-set predictions named `test_set_labels.csv`
- output format matching `annotations/training_set_labels.csv`
- a poster or oral-presentation narrative explaining the approach

The repository keeps the final CSV output under `HELLOWORLD/results/test_set_labels.csv`.
