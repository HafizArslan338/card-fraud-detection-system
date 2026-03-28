# Credit Card Fraud Detection 2023: A Business Intelligence Approach

**Module:** COMP11122 – Data Mining & Business Intelligence 2025/26 T2
**University:** University of the West of Scotland (UWS)

## 🚀 Project Overview

This project builds a fraud detection system using **Orange Data Mining** and **GPU-accelerated Python**.
The model was trained on **550,000+ transactions** to identify patterns of fraudulent behaviour using PCA-transformed features (V1–V28).

## 🛠️ Tech Stack & Hardware

* **Software:** Orange 3.x, Python 3.10+, PyTorch, XGBoost
* **Hardware:** NVIDIA GeForce RTX 4060 (CUDA enabled)
* **Dataset:** Credit Card Fraud Detection Dataset 2023 (Kaggle)

## 📊 Key Results

We compared three models using **10-fold cross-validation**:

| Model                    | Accuracy | Recall (Fraud) | Status    |
| ------------------------ | -------- | -------------- | --------- |
| XGBoost (GPU)            | **1.00** | **1.00**       | Champion  |
| Neural Network (PyTorch) | 0.96     | 0.94           | Optimized |
| Logistic Regression      | 0.96     | 0.93           | Baseline  |

## 🧠 Data Mining Insights

* **Feature Importance:** V14 and V10 were the strongest fraud indicators
* **Association Rules:** Example — IF V14 is Low AND V12 is Low → Fraud (98% confidence)

## 👥 Group Members

* Hafiz Muhammad Arslan Razzaq
* Muhammad Muzammil Jabbar
* Muhammad Kashan
* Muhammad Zubair Anwar
* Muhammad Mohsin Ali

## 📌 Conclusion

The **XGBoost GPU model** achieved the best performance with perfect fraud detection recall.
The results demonstrate that combining **Business Intelligence tools** with **GPU-accelerated machine learning** can produce highly reliable fraud detection systems for real-world financial applications.


## 📂 Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset 2023** from Kaggle.

* Total Transactions: 550,000+
* Fraud Cases: Highly imbalanced dataset
* Features: 28 PCA-transformed variables (V1–V28)
* Target Variable: Class (0 = Normal, 1 = Fraud)

Dataset link:
https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

---

## 🔄 Project Workflow

The project follows a structured data mining and machine learning workflow:

1. Data loading and preprocessing
2. Handling class imbalance
3. Feature analysis using Orange Data Mining
4. Model training:

   * Logistic Regression
   * Neural Network (PyTorch)
   * XGBoost (GPU)
5. Model evaluation using 10-fold cross-validation
6. Feature importance analysis
7. Association rule mining
8. Performance comparison and model selection

---

## ▶️ How to Run the Project

### 1. Clone the repository


git clone https://github.com/your-username/your-repository-name.git


### 2. Navigate to project folder


cd your-repository-name


### 3. Install dependencies


pip install -r requirements.txt


### 4. Run the training script


python train_model.py

### 5. (Optional) Open Orange workflow

Open the `.ows` file using Orange Data Mining to view the visual data mining workflow.


## 📁 Project Structure

├── data/
├── notebooks/
├── models/
├── orange_workflow/
├── train_model.py
├── requirements.txt
└── README.md

