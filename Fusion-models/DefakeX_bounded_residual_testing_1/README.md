# DefakeX_bounded_residual_testing_1

## Training Notebook

`DefakeX_bounded_residual_testing_1.ipynb`

## Evaluation Results

- Tested bounded residual variants including max residual values such as `0.5`, `1.0`, `1.5`, and `2.0`
- Example validation result for max residual `2.0`: score `0.8787`, macro F1 `0.9285`, real recall `0.9288`, fake recall `0.9524`, AUC `0.9794`
- Example validation result for max residual `1.5`: score `0.8774`, macro F1 `0.9273`, AUC `0.9793`

## Datasets Used

- Uses cached heterogeneous multi-view fusion features from the frequency-model checkpoint dataset.
- The notebook evaluates cached train/validation/test splits rather than loading raw images.

## Fusion Technique

- Tests bounded residual fusion heads with a controlled maximum residual value.
- The residual contribution is clipped/scaled so the fusion model can adjust expert outputs without fully overriding them.

## Why This Technique Is Used

Bounded residuals are used to add correction capacity while keeping the fused prediction anchored to the expert signals.
