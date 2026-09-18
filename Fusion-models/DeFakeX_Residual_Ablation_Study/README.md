# DeFakeX_Residual_Ablation_Study

## Training Notebook

`DeFakeX_Residual_Ablation_Study.ipynb`

## Evaluation Results

- Residual scales compared: `0.00`, `0.05`, `0.25`
- Best validation rank in the displayed champion summary: residual scale `0.25`
- Validation at residual `0.25`: validation score `0.8834`, macro F1 `0.9308`, worst scenario `0.9337`
- Mixed held-out at residual `0.25`: accuracy `0.9437`, macro F1 `0.9407`
- FF++ at residual `0.25`: accuracy `0.9522`, macro F1 `0.9086`
- CelebDF at residual `0.25`: accuracy `0.7274`, macro F1 `0.6872`

## Datasets Used

- Cached feature file: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/heterogeneous_multiview_features.pt`
- Available cached splits shown in the notebook: train, validation, mixed test, FF++ test, CelebDF test, and phone test.

## Splits

- Train: `206,711`
- Validation: `44,349`
- Mixed test: `16,599`
- FF++ test: `26,276`
- CelebDF test: `30,000`
- Phone test: `264`

## Fusion Technique

- Standalone residual ablation over cached multi-view features.
- Trains and evaluates fusion variants with different residual scales while holding calibration, normalization, seed, sampler, optimizer, scheduler, early stopping, and validation objective constant.

## Why This Technique Is Used

The residual ablation isolates whether adding bounded residual capacity improves the fusion model beyond the cached expert features without changing the rest of the experimental protocol.
