# deepfake-detection-model-v2

## Training Notebook

`training_notebook.ipynb`

## Accuracy

- FFT Test Accuracy: `0.8058`
- FFT F1-score: `0.8191`
- Best validation accuracy observed in training logs: `0.8004`

## Datasets Used

- `140k-real-and-fake-faces` (`/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake`)
- `CIFAKE` (`/kaggle/input/cifake-real-and-ai-generated-syn`)
- Inference/demo images: `fft-test-images` (`/kaggle/input/fft-test-images/...`)

## Deepfake Detection Technique

- Builds a merged binary dataset in `/kaggle/working/fft_merged` from:
- Real/Fake splits in the 140k dataset
- Additional fake samples from CIFAKE
- Uses FFT log-magnitude preprocessing plus high-pass filtering to transform images into frequency-domain inputs.
- Computes dataset-level FFT normalization statistics from the training split.
- Trains a ResNet18-based binary classifier with:
- ImageNet-pretrained initialization
- Modified first convolution for FFT input
- Cross-entropy loss + Adam optimizer

