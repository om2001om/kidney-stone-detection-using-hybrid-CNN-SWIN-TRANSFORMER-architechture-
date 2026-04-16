<h1 align="center">🧠 Hybrid CNN–Swin Transformer for Kidney Stone Detection</h1>

<p align="center">
  <img src="[https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)" alt="Python" />
  <img src="[https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)" alt="PyTorch" />
  <img src="[https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)" alt="Kaggle" />
</p>

> **A deep learning project for automated kidney stone detection and classification using a hybrid CNN + Swin Transformer architecture, combining local feature extraction with global contextual understanding.**



---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Training Pipeline](#-training-pipeline)
- [Results](#-results)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualizations](#-visualizations)
- [Future Work](#-future-work)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Kidney stone detection using CT scans is a critical medical imaging task. Manual diagnosis is time-intensive and subject to variability due to similarities between kidney stones, cysts, and tumors.

This project introduces a **hybrid deep learning framework** that:
1. Uses **CNNs** for extracting local texture and edge features.
2. Uses **Swin Transformers** for capturing global spatial dependencies.
3. Combines both using a **cross-attention fusion mechanism**.

### 🎯 Objectives
- Improve diagnostic accuracy.
- Reduce false positives/negatives.
- Enhance model interpretability.
- Provide a scalable clinical solution.

---

## ✨ Key Features

- 🔬 **Hybrid Architecture:** Combines the strengths of CNNs and Swin Transformers.
- 🎯 **Dual Classification:** Supports both binary and multi-class classification.
- 🧠 **Cross-Attention Fusion:** Intelligently merges feature representations.
- 🔁 **Robust Validation:** K-Fold cross-validation for reliable metrics.
- 📉 **Advanced Loss Functions:** Focal Loss implemented to handle class imbalance.
- ⚡ **Efficiency:** Mixed Precision Training (AMP) enabled.
- 📊 **Explainability:** Integrated Grad-CAM & Attention Maps.

---

## 🏗️ Architecture

### 🔹 High-Level Overview

> <img width="1109" height="1136" alt="arcoverpng1" src="https://github.com/user-attachments/assets/e75c1da6-d444-4f69-b88a-b68febf2ca8c" />

The architecture consists of two parallel branches:
* **CNN branch** $\rightarrow$ Captures local features (edges, textures).
* **Swin Transformer** $\rightarrow$ Models global relationships and anatomical context.
* **Fusion module** $\rightarrow$ Merges both feature representations for the final classification.

### 🔹 Core Components
<img width="1993" height="1970" alt="arcmain1" src="https://github.com/user-attachments/assets/59bb6a05-8324-4801-85e5-28487a25860b" />

#### 1. CNN Backbone
- 4 convolutional blocks.
- **Channels:** `64` $\rightarrow$ `128` $\rightarrow$ `256` $\rightarrow$ `512`.
- Extracts fine-grained details like calcification patterns and edges.

#### 2. Squeeze-and-Excitation (SE) Blocks

<img width="3080" height="1709" alt="Diagrams_Kidney_Stone_SE_BLOCK1" src="https://github.com/user-attachments/assets/835bbd2f-5ee0-4929-96e2-ba4354468da0" />

- Channel attention mechanism that enhances clinically relevant features while suppressing noise.

#### 3. Swin Transformer

<img width="3240" height="1773" alt="_Diagrams_Kidney_Stone_swin_transformer_block1" src="https://github.com/user-attachments/assets/bba203e9-346d-48e1-8ba1-cc8c1b3f836c" />

- Patch-based transformer utilizing hierarchical attention to capture the global anatomical context of the CT scans.

#### 4. Cross-Attention Fusion
- **CNN features:** Act as Queries.
- **Transformer features:** Act as Keys & Values.
- Enables interaction between local and global features, leading to context-aware predictions.
- **Final Feature Vector:** `1280` dimensions.

---

## 📂 Datasets

This project uses publicly available CT scan datasets for kidney stone detection and classification.

---

### 🔹 Binary Classification Dataset (Stone vs Non-Stone)

- 📌 **Axial CT Imaging Dataset (Kidney Stone Detection)**
- 🔗 Link: https://www.kaggle.com/datasets/orvile/axial-ct-imaging-dataset-kidney-stone-detection

**Description:**
- Contains CT scan images labeled as **Stone** and **Non-Stone**
- Includes **3,364 original images** and **35,000+ augmented images**
- Collected from medical centers in Iraq and annotated by professionals :contentReference[oaicite:0]{index=0}

---

### 🔹 Multi-Class Dataset (Normal, Cyst, Tumor, Stone)

- 📌 **CT Kidney Dataset: Normal-Cyst-Tumor-Stone**
- 🔗 Link: https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone

**Description:**
- Contains **12,446 CT images**
- Classes:
  - Normal
  - Cyst
  - Tumor
  - Stone
- Widely used for multi-class kidney disease classification :contentReference[oaicite:1]{index=1}

---

### ⚠️ Notes

- Both datasets are sourced from **Kaggle**
- Suitable for:
  - Medical image classification
  - Deep learning research
  - Computer vision tasks
- May contain **class imbalance and dataset bias**, as noted in the research paper

---

### ⚠️ Dataset Challenges Addressed
- **Class imbalance:** (Stones represented only ≈11% of the multi-class data).
- **Scanner variability:** Addressed via robust normalization.
- **Dataset bias:** Mitigated through aggressive augmentation and stratification.

---

## 📁 Project Structure

```bash
.
├── notebooks/
│   ├── cnn-swint-transformer-dataset-2-class.ipynb
│   ├── k-fold-dataset-2-4-class-kidney-stone.ipynb
│
├── models/             # Saved model weights
├── data/               # Dataset directory (gitignored)
├── results/            # Evaluation metrics and plots
├── utils/              # Helper functions and data loaders
├── README.md
└── requirements.txt
```
## 🔄 Training Pipeline

### 1. Preprocessing & Augmentation
- **Resize:** `224 × 224`
- **Normalization:** ImageNet statistics
- **Augmentation:** Random Horizontal Flip, Rotation (±15°), Color Jitter, CutMix.

### 2. Training Configuration
- **Optimizer:** AdamW
- **Loss Function:** Focal Loss + Label Smoothing ($\epsilon = 0.1$)
- **Scheduler:** Warmup with Cosine Annealing

### 3. Validation Strategy
- Stratified K-Fold Cross Validation.

---

## 📊 Results

### 🔹 Binary Classification Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 96.67% |
| **Precision** | 0.9845 |
| **Recall** | 0.9696 |
| **F1 Score** | 0.9573 |
| **ROC-AUC** | 0.9807 |

### 🔹 Multi-Class Classification Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 100.0% |
| **Precision** | 1.000 |
| **Recall** | 1.000 |
| **F1 Score** | 1.000 |

---

## 📈 Visualizations

### Model Performance & Interpretability
> *Training and validation loss/accuracy curves over epochs.*
> <img width="4230" height="1847" alt="loss accD1" src="https://github.com/user-attachments/assets/c66f9f43-d2ae-49c7-a782-25d14d928d0e" />
<img width="4083" height="1934" alt="loss accD2" src="https://github.com/user-attachments/assets/bb47463f-a108-4944-9a60-d9dca53674ec" />



<img width="3955" height="1902" alt="confusion matrix" src="https://github.com/user-attachments/assets/f3387fbb-c5e7-4b9f-827d-0d8da54c352b" />

> *Performance breakdown across all classes.*

### Explainability Methods
To validate clinical relevance, the model's decision-making process is visualised using:

<img width="832" height="556" alt="grad cam" src="https://github.com/user-attachments/assets/a1b95909-b284-4209-8c1d-ad1dffdd069c" />

> *Grad-CAM highlights regions of interest utilised by the CNN backbone.*

<img width="1096" height="549" alt="cross attention visualisation" src="https://github.com/user-attachments/assets/c4ce836d-db3d-4f50-8cef-842e146041b4" />

> *Cross-Attention Maps demonstrate how local and global features interact.*

---

## 🔮 Future Work

- [ ] Multi-centre dataset validation to test out-of-distribution generalization.
- [ ] Integration of full 3D CT scan volumes.
- [ ] Development of a real-time inference system/UI.
- [ ] Prediction of stone size and composition.

---

## ⭐ Acknowledgements

*Add any credits to dataset creators, inspiration papers, or mentors here.*
