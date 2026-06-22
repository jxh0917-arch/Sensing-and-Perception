# Ablation Plan

This page separates measured results from proposed comparisons. The current repository contains one completed experiment and a structured plan for future ablations.

## Completed Experiment

| ID | Variant | Global branch | Local branch | Fusion | Validation accuracy | Macro F1 | Status |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| E1 | Dual-branch final model | R3D-18 on 16 person crops | SmallArmCNN on first/last arm crops | Fixed weighted late fusion, `0.7/0.3` | 99.05% | 0.9904 | Measured |

## Planned Comparisons

| ID | Variant | Hypothesis | Expected diagnostic value |
| --- | --- | --- | --- |
| A1 | R3D-18 only | Whole-body temporal motion explains most class separation. | Quantifies the contribution of the global video branch. |
| A2 | SmallArmCNN only | Arm/tool crops are useful but insufficient without full trajectory context. | Tests whether local cues alone can distinguish manipulation-heavy classes. |
| A3 | Dual branch, `0.8/0.2` fusion | A smaller arm weight may reduce noise when pose estimates are uncertain. | Tests sensitivity to the chosen fusion rule. |
| A4 | Dual branch, `0.6/0.4` fusion | A larger arm weight may improve tool-heavy classes. | Tests whether local evidence should be emphasised more strongly. |
| A5 | Arm crops from all 16 sampled frames | More local temporal evidence may reduce direction/tool ambiguity. | Tests whether the arm branch is limited by sparse temporal sampling. |
| A6 | Learned fusion layer | A learned combiner may adapt class-wise branch weighting. | Compares fixed interpretability with adaptive performance. |
| A7 | Optical-flow or trajectory cue | Backward and diagonal-backward errors are direction-sensitive. | Tests whether explicit motion direction improves the remaining weak classes. |

## Reporting Protocol

Each ablation should use the same train/validation/test split, random seed, frame sampling policy, and evaluation script unless the ablation explicitly changes one of those components.

Recommended reporting fields:

| Field | Purpose |
| --- | --- |
| Variant ID | Stable experiment reference |
| Changed component | Architecture, preprocessing, fusion, or training setting |
| Validation accuracy | Main headline metric |
| Macro F1 | Class-balanced metric |
| Weak classes | Classes with precision or recall below `1.0` |
| Runtime or memory note | Practical cost of the change |
| Interpretation | Whether the result supports the hypothesis |

## Interpretation Standard

Ablations should not only report whether a number improves. They should explain which evidence source is responsible for the change:

- global temporal motion should help direction and walking classes
- local arm/tool crops should help drilling, polishing, pickup, and put-down classes
- fusion changes should be evaluated against the error pattern in `docs/error_analysis.md`

This makes the project easier to discuss as a research workflow rather than only a high-score submission.
