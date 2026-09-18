# final-frequency-branch-experiment

## Training Notebook

`frequency-branch-notebook.ipynb`

## Evaluation Results

- Best checkpoint epoch: `7`
- Validation: accuracy `0.8931`, macro F1 `0.8930`, real F1 `0.8892`, fake F1 `0.8968`
- Test: accuracy `0.8896`, macro F1 `0.8895`, real F1 `0.8853`, fake F1 `0.8936`
- Worst-source F1 shown in the full evaluation: `0.8235`

## Datasets Used

- StyleGAN real/fake faces: `/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/train`
- Flickr30k real images: `/kaggle/input/datasets/adityajn105/flickr30k/Images/flickr30k_images`
- DeepDetect fake images: `/kaggle/input/datasets/ayushmandatta1/deepdetect-2025/ddata/train`
- OpenFake parquet parts:
  - `/kaggle/input/datasets/kashirhanif/openfake-part-1`
  - `/kaggle/input/datasets/kashirhanif/openfake-part-2`
  - `/kaggle/input/datasets/kashirhanif/openfake-part-3`
  - `/kaggle/input/datasets/kashirhanif/openfake-part-4`
- Resume checkpoint: `/kaggle/input/datasets/frequency-model-checkpoint/best_model.pth`

## Splits

- Train: `189,396`
- Validation: `25,252`
- Test: `37,884`
- Training balance shown in the notebook: `97,125` real and `92,271` fake

## Deepfake Detection Technique

- Final frequency branch built around FFT-derived inputs rather than raw RGB pixels.
- Extracts selected OpenFake parquet images to JPEG once, then trains from JPEG paths to avoid repeated parquet loading and memory pressure.
- Builds train/validation/test splits from StyleGAN, Flickr, DeepDetect, and OpenFake.
- Uses an EfficientNet-B3 binary classifier with a single-logit output.
- Uses FFT preprocessing with clipping, a forced positive class weight, AdamW, cosine annealing, AMP, gradient clipping, and macro F1 for best-checkpoint selection.

## Why This Technique Is Used

The notebook explicitly states that training all fake sources together is preferred over phase-wise training because a single mixed run encourages one real-vs-fake boundary across GAN and diffusion sources. The frequency representation is used to emphasize spectral artifacts that may be less visible in pixel space.
