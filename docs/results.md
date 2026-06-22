# Results

## Validation Summary

The current validation report in `HELLOWORLD/results/test_report.txt` records:

| Metric | Value |
| --- | ---: |
| Overall accuracy | 0.9905 |
| Macro precision | 0.9917 |
| Macro recall | 0.9905 |
| Macro F1 | 0.9904 |
| Micro F1 | 0.9905 |

The validation split contains 7 examples per class across 30 classes.

## Error Pattern

Most classes reached perfect validation precision and recall. The visible confusion is concentrated around backward and diagonally backward motions:

- `MoveBackwardsWhileDrilling`
- `MoveBackwardsWhilePolishing`
- `MoveDiagonallyBackwardLeftWithDrill`
- `MoveDiagonallyBackwardLeftWithPolisher`

This is plausible because these classes differ by subtle direction and tool cues. The arm branch is intended to help with the tool cue, while the R3D branch handles temporal direction.

## Test Output

The final generated test-set prediction file is:

```text
HELLOWORLD/results/test_set_labels.csv
```

A detailed file with probabilities is also available locally:

```text
HELLOWORLD/results/test_set_labels_detailed.csv
```

The detailed file is useful for confidence analysis but is not the required coursework submission format.

## Interpretation

The high validation score suggests that the combined representation fits the labelled development data well. For a research presentation, the stronger claim is not simply the number, but the structured argument: the model decomposes industrial activity recognition into global motion and local arm/tool evidence, then recombines them through an explicit fusion rule.
