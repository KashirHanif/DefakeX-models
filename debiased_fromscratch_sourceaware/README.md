## Accuracy

- Final test accuracy (`TEST (EPOCH 11 MODEL)`): `0.8651`
- Best validation accuracy shown: `0.8626` (Epoch 11)

## Datasets Used

- `140k-real-and-fake-faces` (StyleGAN)
- `datasets/wish096/realvsfake-81k-by-wish/RealVsFake/RealVsFake` (Wish RealVsFake data)
- `datasets/adityajn105/flickr30k/Images/flickr30k_images` (Flickr)
- `datasets/ayushmandatta1/deepdetect-2025/ddata/train` (DeepDetect)
- Checkpoint input (for load/eval/resume path in notebook):
- `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/latest_resnet18_fft_sourceaware.pth`
- Output checkpoint:
- `/kaggle/working/latest_resnet18_fft_sourceaware.pth`
- `/kaggle/working/best_resnet18_fft_sourceaware.pth`

## Deepfake Detection Technique

- From-scratch FFT-based source-aware ResNet18 training pipeline.
- Builds an index with explicit source buckets to carefully balance the dataset:
  - `stylegan_real`: 50000
  - `stylegan_fake`: 50000
  - `wish_real`: 1000
  - `wish_fake`: 1000
  - `flickr_real`: 30000
  - `deepdetect_fake`: 30000
- Uses a source-aware custom sampling approach (`BalancedSourceBatchSampler` and `WeightedRandomSampler`) to ensure balanced coverage of each domain within an epoch.
- Applies FFT magnitude transform (`fft_transform`) and random frequency masking (`frequency_mask`).
- Generates heavy data augmentations (`make_phone_like`) such as scaling, median filtering, contrast adjustments, unsharp mask, and JPEG compression simulating phone camera sensors to improve robustness.
