# spatial_branch_with_linear_probe

## Training Notebook

`spatial_branch_with_linear_probe.ipynb`

## Evaluation Results

- Linear probe full FF++ test: accuracy `0.9353`, macro F1 `0.8856`, real F1 `0.8101`, fake F1 `0.9610`, AUC `0.9781`
- Linear probe CelebDF: accuracy `0.5974`, macro F1 `0.5970`, real F1 `0.5833`, fake F1 `0.6106`, AUC `0.7729`
- Comparison table shows original spatial head on CelebDF: accuracy `0.7513`, macro F1 `0.6918`, AUC `0.7641`

## Datasets Used

- FaceForensics++ C23 extracted faces
- CelebDF-v2 image dataset
- LFW real faces
- Phone images
- Spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model_spatial.pth`

## Splits

- FF++ train: `128,477`
- FF++ validation: `27,493`
- FF++ test: `26,276`
- CelebDF test: `30,000`

## Spatial Technique

- Starts from a trained spatial EfficientNet-B3 branch and extracts embeddings.
- Trains/evaluates a linear probe over spatial features.
- Compares the probe against the original spatial head on FF++ and CelebDF.

## Why This Technique Is Used

The linear probe tests how much class information is already present in the spatial embeddings and whether a simpler decision head improves cross-dataset behavior.
