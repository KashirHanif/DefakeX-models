# best-freq-model-v4 (deepfake-v7)

## Training Notebook

`training-notebook.ipynb`

## Accuracy

## Datasets Used

- `140k-real-and-fake-faces` (`/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake`)
- `ffhq-1024x1024` (`/kaggle/input/ffhq-1024x1024/images1024x1024`)
- `deepdetect-2025` (`/kaggle/input/deepdetect-2025/ddata`)

## Deepfake Detection Technique

- Frequency-domain deepfake detection using FFT-based features extracted from image luminance.
- Builds multi-scale FFT log-magnitude maps with ring normalization (`fft_multiscale_ringnorm_tensor`).
- Uses a custom `FFTDomainDataset` and a domain-balanced `WeightedRandomSampler` to reduce dataset/domain bias.
- Model is a ResNet18-based embedding network (`FFTResNetEmbed`) trained with:
- Label-smoothed cross-entropy (`LabelSmoothingCE`)
- Supervised contrastive loss (`supervised_contrastive_loss`)
- Includes image degradation augmentations (e.g., JPEG recompression / crop pipeline) to improve robustness.

