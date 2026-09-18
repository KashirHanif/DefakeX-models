# fusion_notebook

## Training Notebook

`fusion_notebook.ipynb`

## Evaluation Results

No completed metric outputs were present in the saved notebook.

## Datasets Used

- OpenFake manifest: `/kaggle/input/datasets/frequency-model-checkpoint/of_manifest.csv`
- Frequency checkpoint: `/kaggle/input/datasets/frequency-model-checkpoint/best_model.pth`
- Spatial checkpoint: `/kaggle/input/datasets/spatial-v2-checkpoint/best_model_spatial_v2.pth`
- StyleGAN, Flickr, DeepDetect, FF++, phone images, and CelebDF are referenced in the notebook.

## Fusion Technique

- Combines a frequency-branch logit and a spatial-branch logit with a small `FusionMLP(2 -> 32 -> 16 -> 1)`.
- The notebook describes the fusion head as learning when to trust spectral evidence, spatial evidence, or both.
- Phone images are included as a conflict case, while CelebDF is reserved as a held-out test.

## Why This Technique Is Used

The two branches target different evidence: the frequency branch targets spectral AI-generation artifacts, while the spatial branch targets face manipulation artifacts. A fusion head is used to learn how to combine those signals when they agree or conflict.
