# spatial_v2_testing_mobile

## Training Notebook

`spatial_v2_testing_mobile.ipynb`

## Evaluation Results

- Reuses the spatial v2 training/evaluation flow with logged best validation macro F1 reaching `0.9259` in the saved outputs.
- The notebook also loads frequency and spatial checkpoints to evaluate phone images and CelebDF across branches.

## Datasets Used

- FaceForensics++ C23 extracted faces split root: `/kaggle/input/datasets/gradientvoyager/faceforensics-c23-extracted-faces-100k/dataset_processed_split`
- LFW deep-funneled real faces
- CelebDF-v2
- Phone images: `/kaggle/input/datasets/kashirhanif/phone-images`
- Frequency checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model.pth`
- Spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model_spatial_v2.pth`

## Splits

- Spatial v2 train: `138,227`
- Validation: `28,675`
- Test: `28,122`
- CelebDF held-out test: `23,000`

## Spatial Technique

- Spatial v2 EfficientNet-B3 branch trained on FF++ plus LFW real faces.
- Includes a later cross-evaluation section comparing frequency and spatial branches on phone images and CelebDF.

## Why This Technique Is Used

The notebook checks whether the spatial v2 branch improves real-phone handling and how it compares with the frequency branch on held-out mobile and CelebDF data.
