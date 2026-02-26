# best-freq-model-v5 (deepfake-v8)

## Training Notebook

`training-notebook.ipynb`

## Accuracy

- Reported best validation accuracy after resume: `0.8519` (shown when checkpoint is loaded).
- Additional logged validation accuracies in current run: `0.8345` (epoch 5), `0.6968` (epoch 6).

## Datasets Used

- `140k-real-and-fake-faces` (`/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake`)
- `ffhq-1024x1024` (`/kaggle/input/ffhq-1024x1024/images1024x1024`)
- `deepdetect-2025` (`/kaggle/input/deepdetect-2025/ddata`)
- Checkpoint input for resume:
- `frequency-model-checkpoint` (`/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/freq_v5_epoch2_checkpoint.pt`)

## Deepfake Detection Technique

- Frequency-domain classifier with handcrafted spectral feature construction before CNN classification.
- Builds feature tensors from:
- FFT magnitude (`fft_mag`)
- Ring normalization
- RAPS-like map (`compute_raps_map`)
- High-frequency residual (`high_freq_residual`)
- Uses a ResNet18-based classifier (`FreqNet`) on frequency features.
- Uses weighted sampling and label-smoothed cross-entropy for training.
- Supports checkpoint resume (`model_state`, `optimizer_state`, `epoch`, `best_acc`).

