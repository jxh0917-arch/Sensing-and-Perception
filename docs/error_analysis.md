# Error Analysis

## Validation Error Focus

![Validation error focus](assets/validation_error_focus.png)

Most validation classes achieved perfect precision and recall. The non-perfect classes are concentrated around backward and diagonal-backward motions:

- `MoveBackwardsWhileDrilling`
- `MoveBackwardsWhilePolishing`
- `MoveDiagonallyBackwardLeftWithDrill`
- `MoveDiagonallyBackwardLeftWithPolisher`

This pattern is plausible for the dataset: these classes differ by subtle movement direction and tool context. The model generally separates tool categories well, but the remaining errors suggest that direction estimation under similar body poses is still the most fragile part of the pipeline.

## What the Errors Suggest

The error pattern supports three follow-up directions:

- Add a trajectory or optical-flow cue to strengthen direction-sensitive recognition.
- Use arm crops from all sampled frames rather than only first and last frames.
- Run ablations for fusion weights to test whether the arm branch should contribute differently for tool-heavy classes.

## Caveat

This analysis is based on the available validation report. The repository does not include raw validation predictions or the hidden test labels, so this page does not claim a full confusion-matrix reconstruction.
