# SkinRisk++: Ordinal Skin-Lesion Severity Staging

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official implementation and manuscript for the clinical triage framework **SkinRisk++**. The system maps the seven diagnostic categories of the HAM10000 dataset onto a 5-stage ordinal severity scale (Normal to Critical) to assist frontline primary-care physicians.

## 🚀 Overview

SkinRisk++ predicts lesion severity using a heterogeneous ensemble of three deep learning backbones:
* **EfficientNet-B4**
* **EfficientNet-B5**
* **ConvNeXt-Base** (ImageNet-22K Pretrained)

Unlike traditional pipelines that append patient metadata at the very end, SkinRisk++ utilizes **per-backbone metadata branches**. This allows demographic features (Age, Biological Sex, and Anatomical Site) to interact directly with image features before the soft-voting stage. 

To ensure clinical trust, the system implements **Fused Class-Activation Mapping (Grad-CAM)**, combining the spatial explanations from all three networks into a single, highly localized saliency map.

## 📊 Performance Metrics

The ensemble was evaluated on 1,503 held-out test images from the HAM10000 dataset:
* **Weighted F1 Score:** 86.20%
* **Macro AUC-ROC:** 97.11%
* **Accuracy:** 85.36%
* **Expected Calibration Error (ECE):** 0.007 (7-fold reduction in over-confidence)
* **Clinical Gain:** Achieved a +18.7 percentage point F1 increase for Stage 2 (Actinic Keratosis) compared to a two-model baseline.

## 📁 Repository Structure

```text
├── data/                                 # HAM10000 dataset files and metadata (ignored in Git)
├── figures/                              # High-resolution PDF/JPG plots and Grad-CAM visualizations
│   ├── fig_architecture.pdf              # Ensemble architecture diagram
│   ├── fig_gradcam_5column_ensemble.pdf  # Fused spatial explainability maps
│   ├── fig_cm_grid.pdf                   # Confusion matrix and ordinal error distribution
│   └── fig_sens_spec.pdf                 # Per-class sensitivity and specificity
├── SkinRisk.ipynb                        # Main Jupyter Notebook for training, evaluation, and CAM generation
├── skinrisk_final_config.json            # Hyperparameter configurations and dataset mappings
├── final_classification_report.csv       # Tabular output of precision, recall, and F1 per class
└── README.md
```



🛠️ Requirements & Installation
Clone the repository:

Bash
   git clone [https://github.com/blazingarrows1525/SkinRisk-Ensemble]
   cd SkinRisk-Ensemble
Install the required dependencies:

Bash
   pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn
🧠 Model Weights
Due to GitHub's file size limits, the trained model weights (.pth files) are hosted externally.

Download the ensemble weights here: [https://drive.google.com/file/d/1AJemu5Ck3C31aJM571qoVxhaIWgBwo8t/view?usp=drive_link]

Place the downloaded .pth files (convnext ensemble pth file - https://drive.google.com/file/d/1AJemu5Ck3C31aJM571qoVxhaIWgBwo8t/view?usp=drive_link
b5 best pth - https://drive.google.com/file/d/14fN4wondgGgUQm8d5L-2KTODbGJRhIRG/view?usp=drive_link
b4 - https://drive.google.com/file/d/1Dnq7s5pHbGYuoPNyD80_TcG9BONNdoKm/view?usp=drive_link.) 
in the root directory before running inference in the notebook.

📈 Training Details
The network handles the severe 20:1 class imbalance of the HAM10000 dataset using:

Loss Function: Class-weighted cross-entropy combined with label smoothing (ϵ=0.1).

Augmentation: Joint image-metadata Mixup (λ∼Beta(0.3,0.3)) and CutMix.

Optimization: AdamW optimizer with CosineAnnealingWarmRestarts and gradient clipping.

Inference Strategy: Exhaustive 2-simplex weight sweep resulting in optimal soft-voting weights of (0.30, 0.30, 0.40).

📄 Citation
If you find this code or methodology useful in your research, please consider citing our work:

Code snippet
@article{trivedi2026skinrisk,
  title={A heterogeneous three-model ensemble with per-backbone metadata fusion and fused class-activation mapping for ordinal skin-lesion severity staging},
  author={Trivedi, Harsh and Kumar, Shiwank and Ranjan, Pranav and A, Sasikumar},
  journal={Turkish Journal of Electrical Engineering & Computer Sciences},
  year={2026}
}
