# fusion_v2(2)

## Training Notebook

`fusion_v2(2).ipynb`

## Evaluation Results

- Validation-selected threshold: `0.8625`
- Validation: accuracy `0.9476`, macro F1 `0.9299`, real recall `0.9384`, fake recall `0.9504`, AUC `0.9816`
- Mixed held-out test: accuracy `0.9411`, macro F1 `0.9381`, AUC `0.9907`
- Pure FF++ test: accuracy `0.9524`, macro F1 `0.9091`, AUC `0.9809`
- CelebDF test: accuracy `0.7111`, macro F1 `0.6750`, AUC `0.7398`
- Phone held-out test: accuracy/real recall `0.9470`

## Datasets Used

- Frequency checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model.pth`
- Spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model_spatial.pth`
- Auxiliary spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/spatial_auxiliary_multitask_head.pth`
- StyleGAN, Flickr, DeepDetect, FaceForensics++ C23 split faces, phone images, CelebDF, and OpenFake parts.

## Splits

- Train: `206,711`
- Validation: `44,349`
- Mixed test: `16,599`
- FF++ test: `26,276`
- CelebDF test: `30,000`
- Phone test: `264`

## Fusion Technique

- Heterogeneous multi-view gated fusion over three expert outputs.
- Uses frequency, original spatial, and auxiliary spatial experts with scenario-aware validation.
- Tracks clean real, AI fake, deepfake fake, and phone real scenarios.

## Why This Technique Is Used

The gated fusion design lets the model weight experts differently by scenario, which is useful when a frequency model and a spatial model do not fail on the same kinds of images.
