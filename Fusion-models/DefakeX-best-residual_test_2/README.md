# DefakeX-best-residual_test_2

## Training Notebook

`DefakeX-best-residual_test_2.ipynb`

## Evaluation Results

- The notebook evaluates residual-fusion candidates on validation and original held-out splits.
- It includes threshold sweeps, acceptance gates, uncertainty/coverage analysis, and comparison tables for FF++, CelebDF, phone, and mixed test sets.
- The saved notebook contains extensive rendered outputs; use the in-notebook tables for exact final candidate selection.

## Datasets Used

- Cached heterogeneous multi-view features derived from the frequency, spatial, and auxiliary branches.
- Original held-out splits include mixed test, pure FF++, CelebDF, and phone images.

## Fusion Technique

- Residual fusion candidate testing.
- Evaluates candidate thresholds and residual bounds on the original held-out splits rather than only on validation.
- Tracks macro F1, class recalls, AUC, acceptance gates, and uncertainty coverage.

## Why This Technique Is Used

This notebook checks whether the chosen residual fusion model generalizes beyond validation and whether it passes deployment-style gates such as FF++ recall, CelebDF AUC, and phone real recall.
