# fft_resnet-18-debiased_fromscratch

## Training Notebook

`notebook08a7b32dc4.ipynb`

## Accuracy

- Final test accuracy (`TEST`): `0.9114`
- Best validation accuracy shown: `0.9139`

## Datasets Used

- Dynamically discovered under `/kaggle/input`
- StyleGAN-like folders (`train/real`, `train/fake`)
- `realvsfake-81k-by-wish/RealVsFake/RealVsFake` (Wish RealVsFake data)
- `syntheticeye-diffusion`
- Checkpoint input (for load/eval/resume path in notebook):
- `/kaggle/input/frequency-model-checkpoint/fft_debiased_resnet18_fromscratch.pth`
- Output checkpoint:
- `/kaggle/working/fft_debiased_resnet18_fromscratch_v2.pth`

## Deepfake Detection Technique

- From-scratch FFT-based debiased ResNet18 training pipeline.
- Builds an index with quality/source buckets:
- `real_hq` (StyleGAN real)
- `real_lq` (Wish RealVsFake real)
- `fake_gan` (StyleGAN fake + Wish RealVsFake fake)
- `fake_diff` (SyntheticEye diffusion)
- Creates balanced train/val/test splits from combined real/fake pools.
- Applies FFT magnitude transform (`fft_transform`) and random frequency masking (`frequency_mask`).
- Includes additional image degradation / phone-like augmentation logic (`make_phone_like`) to improve robustness.

