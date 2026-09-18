# spatial-branch-with-exceeding-sota

## Training Notebook

`spatial-branch-with-exceeding-sota.ipynb`

## Evaluation Results

- Training resumes from epoch `5` and continues to epoch `15`
- Logged validation macro F1 around epochs 6-9: `0.8843` to `0.8883`
- The notebook includes FF++ and CelebDF evaluation cells; use the notebook outputs for exact final tables.

## Datasets Used

- FaceForensics++ C23 extracted faces: `/kaggle/input/datasets/gradientvoyager/faceforensics-c23-extracted-faces-100k`
- CelebDF-v2: `/kaggle/input/datasets/pranabr0y/celebdf-v2image-dataset`
- Resume checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model_spatial.pth`

## Splits

- FF++ train: `128,477`
- FF++ validation: `27,493`
- FF++ test: `26,276`
- CelebDF cross-dataset test: `30,000`

## Spatial Technique

- EfficientNet-B3 spatial classifier on RGB face crops.
- Uses ImageNet normalization, geometric/color/JPEG augmentation, and per-manipulation evaluation.
- Uses a weighted sampler to address the FF++ class imbalance.

## Why This Technique Is Used

This spatial branch targets pixel-space face-manipulation artifacts such as texture inconsistencies, blending boundaries, and reenactment artifacts, which are complementary to frequency-domain detection.
