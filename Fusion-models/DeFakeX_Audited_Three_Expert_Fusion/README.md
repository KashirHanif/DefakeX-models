# DeFakeX_Audited_Three_Expert_Fusion

## Training Notebook

`DeFakeX_Audited_Three_Expert_Fusion.ipynb`



## Datasets Used

- StyleGAN, Flickr, DeepDetect, FaceForensics++ C23 split faces, phone images, CelebDF, OpenFake manifest/images.
- Frequency checkpoint: `/kaggle/input/datasets/kashirhanif/frequency-model-checkpoint/best_model.pth`
- Original spatial checkpoint and auxiliary-head checkpoint are loaded as the other two experts.

## Fusion Technique

- Audited three-expert fusion notebook.
- Experts are frequency branch, original spatial branch, and auxiliary spatial head.
- Trains logistic-regression and nonlinear MLP fusion baselines.
- Uses three expert logits plus quality/conflict features.
- Adds checks for broken images, OpenFake path validity, HEIC/HEIF phone images, pure FF++ testing, untouched CelebDF testing, and held-out phone testing.

## Why This Technique Is Used

The notebook is explicitly designed to make the fusion protocol reproducible and auditable, so comparisons are made on the same samples and threshold/model selection penalizes real/fake or scenario collapse.
