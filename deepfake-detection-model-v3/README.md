# deepfake-detection-model-v3

## Training Notebook

`training_notebook.ipynb`

## Accuracy

- `TEST_ID` accuracy: `0.7155`
- `TEST_CROSSGEN` accuracy: `0.9941`

## Datasets Used

- Dynamically discovered under `/kaggle/input` using path search helpers (`find_input_dir`, `find_subdir_by_suffix`, `must_find_dir`)
- StyleGAN-like dataset folders (`train/real`, `train/fake`)
- RealVsFake dataset folders (`Real`, `Fake`)
- `syntheticeye-diffusion-faces`
- Optional `stable-diffusion-face-dataset` (used as `cross_diff` / cross-generation test set when present)
- Inference/demo images: `fft-test-images` (`/kaggle/input/fft-test-images/...`)

## Deepfake Detection Technique

- Builds a multi-source index and converts it to a binary task:
- `real` = stylegan real + RealVsFake real
- `gan` = stylegan fake
- `diff` = RealVsFake fake + SyntheticEye diffusion images
- `cross_diff` = stable-diffusion face images (cross-gen evaluation)
- Uses FFT magnitude preprocessing (`torch.fft.fft2` + `fftshift` + `log1p(abs(.))`) to produce frequency-domain tensors.
- Trains a ResNet18 binary classifier (`weights=None`) on FFT inputs.
- Uses a custom `BalancedBinarySampler` to reduce class imbalance during training.
- Reports both in-domain (`TEST_ID`) and cross-generation (`TEST_CROSSGEN`) metrics.

