# notebookd0881caae1

## Training Notebook

`notebookd0881caae1.ipynb`

## Evaluation Results

- Best fusion head loaded from epoch `14`
- Validation macro F1: `0.8106`
- FF++ test at default threshold: accuracy `0.8170`, macro F1 `0.8081`, AUC `0.9175`
- Pareto threshold: `0.290`
- FF++ test at Pareto threshold: accuracy `0.8314`, macro F1 `0.8199`, real recall `0.8963`, fake recall `0.8005`, AUC `0.9175`

## Datasets Used

- OpenFake manifest: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/of_manifest.csv`
- Frequency checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model.pth`
- Spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model_spatial_v2.pth`
- StyleGAN, Flickr, DeepDetect, FaceForensics++ C23 split faces, phone images, and CelebDF.

## Splits

- Fusion train: `300,282`
- Fusion validation: `48,836`
- Fusion test: `60,216`
- CelebDF held-out test: `23,000`
- Phone split shown: `800` train, `200` validation, `60` test

## Fusion Technique

- Two-expert fusion head using frozen frequency and spatial-v2 branches.
- Trains a compact MLP fusion head over extracted logits.
- Uses threshold sweeping to find a Pareto operating point.

## Why This Technique Is Used

The notebook aims to combine the frequency branch and spatial branch so the fused model can handle cases where one branch is confident and the other is weak or misleading.
