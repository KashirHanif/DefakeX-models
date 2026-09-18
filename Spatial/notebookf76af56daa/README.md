# notebookf76af56daa

## Training Notebook

`notebookf76af56daa.ipynb`

## Evaluation Results

- Best checkpoint noted during training: `best_model_spatial_v2.pth`
- Best validation macro F1 shown during training: `0.9259` at epoch `11`
- FF++ validation after loading best model: accuracy `0.9531`, macro F1 `0.9218`, AUC `0.9815`
- FF++ test: accuracy `0.9586`, macro F1 `0.9359`, real F1 `0.8978`, fake F1 `0.9741`
- CelebDF cross-dataset: accuracy `0.5630`, macro F1 `0.5616`, AUC `0.7274`
- Pareto threshold: `0.791`; FF++ test at this threshold has accuracy `0.9594` and macro F1 `0.9381`

## Datasets Used

- FaceForensics++ C23 extracted faces split root: `/kaggle/input/datasets/gradientvoyager/faceforensics-c23-extracted-faces-100k/dataset_processed_split`
- LFW deep-funneled real faces: `/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled`
- CelebDF-v2: `/kaggle/input/datasets/pranabr0y/celebdf-v2image-dataset/Celeb_V2`

## Splits

- Train: `138,227`
- Validation: `28,675`
- Test: `28,122`
- CelebDF held-out test: `23,000`

## Spatial Technique

- EfficientNet-B3 spatial branch trained from scratch on raw RGB face crops.
- Uses FF++ real/fake manipulations plus LFW real faces for more real-domain diversity.
- Adds heavy augmentation: flips, rotation, brightness/contrast/saturation, grayscale, blur, JPEG compression, sharpening, Gaussian noise, and random erasing.
- Uses macro F1 and Pareto threshold selection.

## Why This Technique Is Used

The notebook states that adding real-source diversity and stronger augmentation is the intended bias fix, preventing the spatial model from learning narrow FF++ real-image statistics.
