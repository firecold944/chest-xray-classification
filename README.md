# Chest X-Ray Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)  
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E=_1.0-orange)](#requirements)  
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~94%25-brightgreen)](#results)  

A clean, easy-to-follow project for binary classification of chest X-rays: NORMAL vs PNEUMONIA. This repository contains training/evaluation scripts, example visualizations, and guidance for improving model performance.

---

Why this repo?
- Practical starting point for pneumonia detection from chest X-rays
- Scripts for training, evaluation and single-image inference
- Visual examples and logs to help debug and present results

Demo images (place these files into docs/images/ before viewing the README to see them inline)

- docs/images/1_normal.png
- docs/images/2_predictions_grid.png
- docs/images/3_training_log.png

Quick preview (images shown large for clarity)

<div align="center">
  <figure>
    <img src="docs/images/1_normal.png" alt="Normal chest X-ray" width="800"/>
    <figcaption><strong>Figure 1:</strong> Example of a NORMAL chest X-ray.</figcaption>
  </figure>
</div>

<div align="center" style="margin-top: 16px;">
  <figure>
    <img src="docs/images/2_predictions_grid.png" alt="Predictions grid" width="1100"/>
    <figcaption><strong>Figure 2:</strong> Grid of model predictions (top captions show Pred: NORMAL / PNEUMONIA).</figcaption>
  </figure>
</div>

<div align="center" style="margin-top: 16px;">
  <figure>
    <img src="docs/images/3_training_log.png" alt="Training log" width="900"/>
    <figcaption><strong>Figure 3:</strong> Example training log showing epoch-by-epoch accuracy and final test accuracy.</figcaption>
  </figure>
</div>

---

Table of contents
- Features
- Repository layout
- Installation
- Quick start (train / evaluate / predict)
- Data conventions
- Results & visualizations
- Best practices & tips
- Interpretability
- Contributing / License / Contact

---

Features
- Data loaders and standard transforms for chest X-ray images
- Support for pretrained backbones (ResNet, EfficientNet, etc.)
- Training loop with metrics logging and checkpointing
- Prediction script for single images and batch inference
- Visualization utilities (prediction grids, Grad-CAM sample script)

Repository layout (recommended)
- data/
  - train/
    - NORMAL/
    - PNEUMONIA/
  - val/
    - NORMAL/
    - PNEUMONIA/
  - test/
    - NORMAL/
    - PNEUMONIA/
- docs/
  - images/                # put images 1_normal.png, 2_predictions_grid.png, 3_training_log.png here
- notebooks/               # exploratory notebooks, visualization, Grad-CAM demos
- models/                  # saved model checkpoints (.pth)
- scripts/
  - train.py
  - evaluate.py
  - predict.py
- requirements.txt
- README.md

---

Requirements

Create a Python virtual environment and install required packages (example):
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Example requirements.txt
```
torch
torchvision
numpy
matplotlib
scikit-learn
tqdm
Pillow
opencv-python
```

---

Quick start

Train (example)
```bash
python scripts/train.py \
  --data-dir data \
  --batch-size 32 \
  --epochs 25 \
  --model-out models/model_best.pth
```

Evaluate
```bash
python scripts/evaluate.py --data-dir data --model models/model_best.pth
```

Predict a single image
```bash
python scripts/predict.py --image docs/images/1_normal.png --model models/model_best.pth
```

Tip: Use --device cuda to speed up training/inference when a GPU is available.

---

Data conventions and preprocessing

- Images grouped into class folders under train/val/test.
- Recommended transforms:
  - Resize to 224x224 (or 256 then center-crop to 224)
  - Normalize using ImageNet mean/std if using pretrained backbones
  - Data augmentation for training: RandomHorizontalFlip, RandomRotation (±10°), slight brightness/contrast jitter
- Watch out for dataset biases: differences in AP vs PA views, presence of markers, image metadata.

---

Results (example)

The example training log (Figure 3) shows high train/val accuracy across epochs; final reported test accuracy ~94% (replace with your model's actual metrics). Use these as a demonstration of workflow rather than guaranteed performance — always validate on held-out clinical data before any practical use.

Suggested metrics to report:
- Accuracy, Precision, Recall, F1-score
- Per-class confusion matrix
- ROC AUC

---

Improving performance — practical tips

- Use pretrained CNN backbones (ResNet, EfficientNet) and fine-tune.
- Freeze early layers initially, then unfreeze for fine-tuning.
- Balance classes with weighted sampling or loss weighting if dataset is imbalanced.
- Use learning rate schedulers (ReduceLROnPlateau, CosineAnnealing).
- Use mixed precision training (torch.cuda.amp) for faster training and lower memory use.
- Perform careful validation: stratify splits by patient ID if available.

---

Interpretability

Add Grad-CAM or other heatmap visualizations to inspect what regions drive model predictions. Example notebook: notebooks/gradcam_demo.ipynb.

A minimal Grad-CAM workflow:
1. Load model and target layer
2. Run forward pass on an image
3. Compute gradients of class output w.r.t. feature maps
4. Aggregate and overlay the heatmap on the original X-ray

---

Contributing

Contributions welcome! Please open issues for bugs or feature requests. If you'd like to contribute code, open a pull request with:
- a clear description of the change
- tests or example usage for new features
- updated documentation if necessary

---

License

This project is distributed under the MIT License. See LICENSE for details.

---

Contact

Author: firecold944  
Repository: https://github.com/firecold944/chest-xray-classification

If you want, I can:
- produce the example scripts (train.py / evaluate.py / predict.py) used by this README,
- create a small notebook with Grad-CAM demonstration,
- or prepare a requirements.txt and CI badge.

Let me know which of those you want next.
