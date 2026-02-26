# deepfake-detection-model-v6

## Training Notebook

`training-notebook.ipynb`

## Accuracy

- Validation accuracy values are logged per epoch.
- Best validation accuracy visible in the saved notebook output (through completed epochs shown): `0.9645`
- Best validation F1 visible in the saved notebook output: `0.9648`
- Notebook output is truncated during epoch 9, so the final run result is not fully recorded in the saved notebook.

## Datasets Used

- Dynamically discovered from `/kaggle/input` using `find_dir_contains(...)`
- StyleGAN-like folders (`train/real`, `train/fake`)
- `celeba` (used as high-quality real images)
- `realvsfake-81k` / `RealVsFake` (used as lower-quality real and GAN fake sources)
- `syntheticeye` (diffusion fake source)
- Output checkpoint path: `/kaggle/working/frequency_debiased_model_v3.pth`

## Deepfake Detection Technique

- Quality-aware, debiased frequency-domain training pipeline.
- Builds a quality-aware index with separate source buckets:
- `real_hq`, `real_lq`, `fake_gan`, `fake_diff`
- Creates quality-balanced train/val/test splits (`create_quality_balanced_splits`) to reduce overfitting to compression/quality cues.
- Uses an `improved_fft_transform` that computes frequency features (including FFT magnitude/phase-derived processing) and normalizes/clamps outputs.
- Applies `frequency_mask_augmentation` for frequency-domain regularization.
- Trains a ResNet18-based detector (`FrequencyDeepfakeDetector`) with focal loss (`FocalLoss`) and validation-based checkpointing.

