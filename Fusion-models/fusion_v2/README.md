# fusion_v2

## Training Notebook

`fusion_v2.ipynb`

## Evaluation Results

No completed metric outputs were present in the saved notebook.

## Datasets Used

- Frequency checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model.pth`
- Original spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model_spatial.pth`
- Auxiliary spatial checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/spatial_auxiliary_multitask_head.pth`
- StyleGAN, Flickr, DeepDetect, FaceForensics++ C23 split faces, phone images, CelebDF, and OpenFake parts are referenced.

## Fusion Technique

- Heterogeneous fusion setup using frequency, spatial, and auxiliary-head checkpoints.
- Builds cached multi-view features in `/kaggle/working/heterogeneous_multiview_features.pt`.
- Separates clean real, AI fake, deepfake fake, and phone real scenarios for fusion training/evaluation.

## Why This Technique Is Used

The notebook is set up to combine complementary experts rather than replace them: frequency evidence covers generation artifacts, spatial evidence covers manipulation artifacts, and the auxiliary head provides another spatial view.
