# Quantitative and Causal Evaluation of Explainability in Brain Tumor MRI Classification

This repository provides the official implementation of the research work submitted to **The Visual Computer (Springer)**.

The code implements a **deep learning pipeline for brain tumor classification with explainable AI (XAI) analysis** using Grad-CAM and segmentation masks for quantitative interpretability evaluation.

---

## Related Manuscript

This code accompanies the manuscript currently submitted to:

**The Visual Computer**

Please cite this paper if you use this code.

> Mukherjee S. , Tewari B. , "Quantitative and Causal Evaluation of Explainability in Brain Tumor MRI Classification", submitted to *The Visual Computer*.

---

## Permanent Research Artifacts

Code repository:
GitHub:  
https://github.com/YOUR_USERNAME/BRISC2025-BrainTumor-XAI


Dataset used:
BRISC 2025 Brain MRI Dataset
https://www.nature.com/articles/s41597-026-06753-y

---

#  Dependencies

Python version:

Python 3.10 – 3.12


## Required libraries:

torch
numpy
pillow
matplotlib
opencv-python
pandas
scipy

## Install dependencies:


pip install -r requirements.txt



# Dataset Structure

The expected dataset directory structure is:


brisc2025/
│
├── classification_task/
│ ├── train/
│ │ ├── glioma/
│ │ ├── meningioma/
│ │ ├── pituitary/
│ │ └── no_tumor/
│ │
│ └── test/
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ └── no_tumor/
│
└── segmentation_task/
└── test/
├── images/
└── masks/


Each segmentation mask corresponds to the tumor region.


# Running the Pipeline

Modify the dataset path inside the script:

```python
DATA_DIR = Path("path_to_dataset/brisc2025")
```

Then run:

python code/brisc_pipeline.py

The pipeline automatically performs:

1. CNN training with multi-seed optimization
2. Classification evaluation
3. Confusion matrix generation
4. Grad-CAM explainability analysis
5. Quantitative XAI metrics
6. Causal occlusion validation

##  Methodology Overview

The pipeline consists of two main components.

### Part I — Brain Tumor Classification

A custom VGG-style CNN is trained to classify MRI images into:

1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

Evaluation metrics include:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. Confusion matrix

Multi-seed training is used to improve robustness.

### Part II — Explainable AI (XAI)

Grad-CAM is applied to the final convolutional block.

Layer used:

model.features[16]

This corresponds to the final convolutional feature map before global pooling.

##  Explainability Metrics

Two quantitative XAI metrics are used.

### ATO (Attention-Tumor Overlap)

Soft overlap between attention map and tumor mask.

ATO = sum(Attention × Mask) / sum(Attention)

### TCR (Tumor Coverage Ratio)

Binary overlap between thresholded attention and tumor region.

TCR = sum(BinaryAttention × Mask) / sum(Mask)

Thresholds used:

0.3
0.5
0.7

## Causal Validation

To verify that the model truly relies on tumor regions, an occlusion test is performed.

Two tests:

1. Tumor occlusion
2. Background occlusion

If tumor occlusion changes prediction significantly, the explanation is considered causally valid.

##  Statistical Analysis

The pipeline includes statistical significance testing using:

Kruskal-Wallis Test

Used to evaluate differences across:

1. Tumor types
2. MRI planes
3. Prediction correctness
