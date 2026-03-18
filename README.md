# Credit Card Fraud Detection 2023: A Business Intelligence Approach
[cite_start]**Module:** COMP11122 - Data Mining & Business Intelligence 2025/26 T2 [cite: 85]
[cite_start]**University:** University of the West of Scotland (UWS) [cite: 1]

## 🚀 Project Overview
[cite_start]This project implements a high-performance fraud detection system using a hybrid methodology of **Orange Data Mining** and **GPU-accelerated Python**. [cite_start]We analyzed over 550,000 transactions to identify a "digital fingerprint" for fraud, focusing on hidden PCA-transformed features (V1-V28)[cite: 188, 189].

## 🛠️ Tech Stack & Hardware
- [cite_start]**Software:** Orange 3.x, Python 3.10+, PyTorch, XGBoost[cite: 181, 247].
- [cite_start]**Hardware Acceleration:** NVIDIA GeForce RTX 4060 GPU (CUDA-enabled).
- [cite_start]**Dataset:** [Credit Card Fraud Detection Dataset 2023 (Kaggle)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)[cite: 195].

## 📊 Key Results
[cite_start]We compared three distinct architectures using 10-fold cross-validation[cite: 271, 312]:
| Model | Accuracy | Recall (Fraud) | Status |
| :--- | :--- | :--- | :--- |
| **XGBoost (GPU)** | **1.00** | **1.00** | [cite_start]**Champion** [cite: 271] |
| **Neural Network (PyTorch)** | 0.96 | 0.94 | [cite_start]Optimized [cite: 271] |
| **Logistic Regression** | 0.96 | 0.93 | [cite_start]Baseline [cite: 271] |

## 🧠 Data Mining Insights
- [cite_start]**Feature Importance:** Discovered that **V14 (f13)** and **V10 (f10)** are the strongest indicators of fraudulent behavior[cite: 289].
- [cite_start]**Association Rules:** Identified high-confidence rules (e.g., IF V14 is Low AND V12 is Low -> Fraud) with **98% confidence**[cite: 305].

## 👥 Group Members
- [cite_start]Hafiz Muhammad Arslan Razzaq (B01827785) [cite: 105]
- [cite_start]Muhammad Muzammil Jabbar (B01831672) [cite: 105]
- [cite_start]Muhammad Kashan (B01830632) [cite: 105]
- [cite_start]Muhammad Zubair Anwar (B01631278) [cite: 105]
- [cite_start]Muhammad Mohsin Ali (B01830477) [cite: 105]