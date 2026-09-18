# mlp+auxillary-head-training

## Training Notebook

`mlp+auxillary-head-training.ipynb`

## Evaluation Results

- Balanced probe test: accuracy `0.9433`, macro F1 `0.9433`, real F1 `0.9427`, fake F1 `0.9438`, AUC `0.9839`
- Full FF++ test: accuracy `0.9472`, macro F1 `0.9027`, real F1 `0.8368`, fake F1 `0.9685`, AUC `0.9783`
- Selected threshold: `0.8075`

## Datasets Used

- FaceForensics++ C23 extracted faces
- LFW deep-funneled faces
- CelebDF-v2 image dataset
- Phone images
- Spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model_spatial.pth`

## Splits

- FF++ train: `128,477`
- FF++ validation: `27,493`
- FF++ test: `26,276`
- LFW split shown: train `9,170`, validation `2,143`, test `1,920`

## Spatial Technique

- Trains an auxiliary MLP head/probe on spatial embeddings from an existing spatial checkpoint.
- Uses FF++ manipulation data, LFW real-face diversity, and additional evaluation on CelebDF/phone data.
- Selects a threshold from validation metrics before full FF++ testing.

## Why This Technique Is Used

The auxiliary head is used to improve spatial-branch decision behavior from embeddings without retraining the full image backbone.
