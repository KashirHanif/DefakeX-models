# fft_resnet-18-debiased

## Training Notebook

`training_notebook.ipynb`

## Accuracy

- `TEST_ID` accuracy: `0.8902`
- `TEST_CROSSGEN` accuracy: `0.9621`
- Best validation accuracy shown: `0.8893`

## Datasets Used

- Dynamically discovered under `/kaggle/input`
- StyleGAN-like folders (`train/real`, `train/fake`)
- RealVsFake dataset folders (`Real`, `Fake`)
- `syntheticeye-diffusion-faces`
- Optional `stable-diffusion-face-dataset` (used for cross-generation evaluation)
- Checkpoint/output paths referenced:
- Input model checkpoint (`MODEL_PATH`) loaded in notebook
- Saved debiased model: `/kaggle/working/fft_binary_resnet18_debiased.pth`

## Deepfake Detection Technique

- Debiased FFT-based ResNet18 binary classifier.
- Builds combined real/fake/cross-generation splits similarly to `deepfake-detection-model-v3`.
- Converts images to frequency-domain inputs with FFT magnitude (`fft_transform`).
- Applies random frequency masking augmentation (`frequency_mask`) to reduce reliance on narrow spectral cues.
- Uses `FFTDebiasedDataset` for training/evaluation and reports validation, in-domain test, and cross-generation test metrics.

