# spatial_branch_training

## Training Notebook

`spatial_branch_training.ipynb`

## Evaluation Results

No completed metric outputs were present in the saved notebook.

## Datasets Used

- FaceForensics++ C23 extracted faces: `/kaggle/input/datasets/gradientvoyager/faceforensics-c23-extracted-faces-100k`
- CelebDF-v2: `/kaggle/input/datasets/pranabr0y/celebdf-v2image-dataset`

## Spatial Technique

- Initial spatial-branch training notebook.
- Uses raw pixel inputs with ImageNet normalization rather than FFT.
- Tracks FF++ manipulation types and includes cross-dataset CelebDF evaluation.
- Uses geometric and color augmentation for the spatial classifier.

## Why This Technique Is Used

The spatial model is intended to detect visible face-manipulation artifacts in pixel space, complementing frequency-domain models that look for spectral generation artifacts.
