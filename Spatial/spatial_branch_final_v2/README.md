# spatial_branch_final_v2

## Training Notebook

`spatial_branch_final_v2.ipynb`

## Evaluation Results

No completed metric outputs were present in the saved notebook.

## Datasets Used

- FaceForensics++ C23 extracted faces split root: `/kaggle/input/datasets/gradientvoyager/faceforensics-c23-extracted-faces-100k/dataset_processed_split`
- LFW deep-funneled real faces: `/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled`
- CelebDF-v2: `/kaggle/input/datasets/pranabr0y/celebdf-v2image-dataset/Celeb_V2`

## Spatial Technique

- Final v2 spatial branch setup for training from scratch.
- Uses FF++ fake manipulation types and FF++ Real plus LFW for real images.
- Keeps phone images and CelebDF as held-out tests according to the notebook plan.
- Uses stronger augmentation and random erasing.

## Why This Technique Is Used

The notebook plan states that LFW real diversity and stronger augmentation are used to reduce real-domain bias, while CelebDF and phone data remain held out for generalization checks.
