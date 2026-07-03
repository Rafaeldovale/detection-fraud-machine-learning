# Credit Card Fraud Detection with Machine Learning

**Data Source:** [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 📍 Quick Navigation
* [Project Overview](#-project-overview)
* [Technical Challenges & Solutions](#️-technical-challenges--solutions)
* [Model Evaluation & Selection](#-model-evaluation--final-selection)
* [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
* [Dataset Information](#-dataset)

---
detection-fraud-machine-learning/
│
├── data/                  # NUNCA envie dados brutos ao GitHub! Guarde-os localmente aqui.
│   ├── raw/               # Dados originais (ex: o CSV do Kaggle)
│   └── processed/         # Dados limpos ou transformados
│
├── notebooks/             # Guarde o seu notebook atual aqui dentro para histórico
│   └── analise_fraude.ipynb
│
├── src/                   # Todo o código Python reutilizável da sua aplicação
│   ├── __init__.py
│   ├── preprocess.py      # Funções de limpeza e transformação de dados
│   └── train.py           # Código para treinar e salvar o modelo
│
├── models/                # Onde salvaremos o modelo treinado (ex: modelo.pkl)
│
├── .gitignore             # Arquivo para dizer ao Git o que NÃO subir
└── README.md

## 📌 Project Overview
This project aims to develop an AI model capable of identifying fraudulent credit card transactions. Using a real dataset (Kaggle), the main challenge is dealing with **extreme data imbalance**, where frauds represent a tiny fraction of total transactions.

## 🛠️ Technologies & Tools
* **Language:** Python 3.11
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Algorithms:** XGBoost, Random Forest, Logistic Regression
* **Version Control:** Git & GitHub

## 🛠️ Technical Challenges & Solutions
Developing a fraud detection model requires a strategic approach to data preparation:

* **Extreme Class Imbalance:** Fraudulent transactions represented only 0.17% of the dataset. I implemented **Random Under-Sampling** to create a balanced 50/50 distribution.
* **Presence of Outliers:** Applied the **Interquartile Range (IQR) Method** to remove extreme values and **Robust Scaling** for features 'Amount' and 'Time'.
* **Model Selection:** While Random Forest is powerful, **Logistic Regression** proved superior in this specific scenario due to proper data cleaning.
* **Prioritizing Recall:** The project was optimized for **Recall**, achieving a 0.97 score, as missing a fraud is more costly than a false alarm.

## 📈 Project Status
- [x] Environment & Repository Setup
- [x] Exploratory Data Analysis (EDA)
- [x] Pre-processing & Cleaning
- [x] Model Training
- [x] Results Evaluation

## 🏆 Model Evaluation & Final Selection
The project compared a linear baseline with a complex ensemble method. **Logistic Regression** emerged as the most effective model.

| Model | True Positives (Frauds Caught) | False Negatives (Missed Frauds) | Recall |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **93** | **3** | **0.97** |
| Random Forest | 89 | 7 | 0.93 |

**Final Decision:** We selected **Logistic Regression** because it minimized the number of missed frauds (only 3 vs 7 from Random Forest).

### 📊 Results Visualization
#### **Winner: Logistic Regression Confusion Matrix**
![Logistic Confusion Matrix](images/cm_logistic_regression.png)

#### **Runner-up: Random Forest Confusion Matrix**
![Random Forest Confusion Matrix](images/cm_random_forest.png)

## 🔍 Exploratory Data Analysis (EDA)
Insights gained during the initial analysis:

* **Data Imbalance**: Visualizing the 0.17% fraud gap.
  ![Class Distribution](images/class_distribution.png)
* **Feature Correlation**: Identifying which PCA variables correlate most with fraud.
  ![Correlation Matrix](images/correlation_matrix.png)
* **Distribution Patterns**: Analysis of transaction amounts and time density.
  ![Amount Distribution](images/amount_density.png)

## 📊 Dataset
The dataset contains transactions made by European cardholders in September 2013. Due to its size (143MB), the CSV file is not included.

[Download Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Note:** Place the `creditcard.csv` file inside the `data/` folder after downloading.