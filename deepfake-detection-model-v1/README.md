# deepfake-detection-model-v1

## Training Notebook

`training_notebook.ipynb`

## Accuracy

- FFT Test Accuracy: `0.8431`
- FFT F1-score: `0.8326`
- Best validation accuracy observed in training logs: `0.8417`

## Datasets Used

- `140k-real-and-fake-faces` (`/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake`)
- `CIFAKE` (`/kaggle/input/cifake-real-and-ai-generated-synthetic-images`)
- Inference/demo images: `fft-test-images` (`/kaggle/input/fft-test-images/...`)
- The notebook also contains an earlier/local dataset-prep section referencing:
- `realvsfake-81k-by-wish`
- `syntheticeye-diffusion-faces`
- `stable-diffusion-face-dataset`

## Deepfake Detection Technique

- Creates a merged binary dataset in `/kaggle/working/fft_merged_balanced_50` using real images from 140k and fake images from both 140k + CIFAKE.
- Converts images to grayscale and computes FFT log-magnitude (`fft_log_magnitude`).
- Applies a high-pass-frequency mask (`high_pass_mask` / `apply_hpf`) to emphasize synthetic artifacts.
- Computes FFT mean/std on training data and normalizes frequency images.
- Trains a modified ResNet18 classifier:
- Starts from ImageNet pretrained weights (`ResNet18_Weights.IMAGENET1K_V1`)
- Replaces `conv1` to accept the FFT representation
- Final FC layer outputs 2 classes (Fake/Real)

