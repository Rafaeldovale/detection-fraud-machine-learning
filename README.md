# detection-fraud-machine-learning

# Credit Card Fraud Detection with Machine Learning

## 📌 Project Overview
This project aims to develop an AI model capable of identifying fraudulent credit card transactions. Using a real dataset (Kaggle), the main challenge is dealing with **extreme data imbalance**, where frauds represent a tiny fraction of total transactions.

## 🛠️ Technologies & Tools
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Algorithms:** XGBoost, Random Forest, Logistic Regression
* **Version Control:** Git & GitHub

## 🚀 Technical Challenges
* **Class Imbalance:** Applying resampling techniques like SMOTE.
* **Performance Metrics:** Focus on **Recall** and **F1-Score** instead of just Accuracy.
* **Feature Engineering:** Analysis of PCA components present in the dataset.

## 📈 Project Status
- [x] Environment & Repository Setup
- [x] Exploratory Data Analysis (EDA)
- [ ] Pre-processing & Cleaning
- [ ] Model Training
- [ ] Results Evaluation

## Tecnologias Utilizadas
- Python 3.11
- Pandas & NumPy
- Scikit-Learn
- Seaborn & Matplotlib

# Key Insights So Far
- **Data Imbalance**: Only 0.17% of transactions are fraudulent, requiring specific evaluation metrics like Recall and Precision.
- **Feature Engineering**: Applied **RobustScaler** to 'Time' and 'Amount' features to handle outliers and align them with PCA-transformed variables (V1-V28).