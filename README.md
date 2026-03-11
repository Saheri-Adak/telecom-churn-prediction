# 📡 Telecom Customer Churn Prediction

> A machine learning project to predict customer churn for a telecom company — comparing **5 models** including an ensemble, achieving **83.5% AUC-ROC** on unseen data with a train-test gap of just **0.032**.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Business Insights](#-key-business-insights)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Learnings](#-key-learnings)

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer costs **5–7x more** than retaining an existing one. This project builds a predictive model to:

- Identify customers who are **likely to churn**
- Understand the **key drivers** behind churn
- Help the business take **proactive retention actions**

---

## 📊 Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Detail |
|---|---|
| Rows | 7,043 customers |
| Features | 20 (demographic, service, billing) |
| Target | Churn (Yes / No) |
| Class Balance | ~26.5% churned, ~73.5% retained |

**Key Features:**
- `tenure` — how long the customer has been with the company
- `Contract` — month-to-month, one year, two year
- `MonthlyCharges` — monthly billing amount
- `TotalCharges` — total amount charged
- `InternetService`, `TechSupport`, `OnlineSecurity` — service subscriptions

---

## 📁 Project Structure

```
Customer Churn Prediction/
│
├── Data/
│   ├── Raw/
│   │   └── Telco-Customer-Churn.csv      # Original IBM dataset
│   └── Processed/
│       └── cleaned_customer_churn_data.csv
│
├── Models/
│   ├── preprocessor.pkl                  # ColumnTransformer pipeline
│   ├── best_xgb_model.pkl               # Tuned Final Model
├── Notebooks/
│   ├── 1_Data_Cleaning.ipynb            # Data cleaning & preprocessing
│   ├── 2_EDA.ipynb                      # Exploratory data analysis
│   ├── 3_Feature_Engineering.ipynb      # Feature encoding & transformation
│   ├── 4_Model_Training.ipynb           # Model training & hyperparameter tuning
│
├── Visuals/                             # All plots and charts
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Data Cleaning
- Removed `customerID` (non-predictive identifier)
- Fixed `TotalCharges` — stored as string in raw data, converted to numeric
- Dropped 11 rows with missing `TotalCharges` (new customers with 0 tenure)
- Consolidated redundant categories (`No phone service` → `No`, `No internet service` → `No`)

### 2. Exploratory Data Analysis
- Profiling report generated with `ydata-profiling`
- Identified strong churn signals in `Contract`, `tenure`, `MonthlyCharges`
- Cohort analysis revealed churn drops sharply after first 12 months
- Service subscription heatmap showed `OnlineSecurity` and `TechSupport` act as retention anchors

### 3. Feature Engineering
Applied a `ColumnTransformer` pipeline with no data leakage:

| Feature Type | Transformation |
|---|---|
| Binary categorical | `OneHotEncoder(drop='if_binary')` |
| Multi-class categorical | `OneHotEncoder(drop='first')` |
| Skewed numerical (all positive) | `PowerTransformer(method='box-cox')` |
| Other numerical | `StandardScaler()` |

### 4. Model Training
Trained and tuned **4 models** using `GridSearchCV` with 5-fold stratified cross-validation, optimizing for `roc_auc`:

- ✅ Random Forest
- ✅ CatBoost
- ✅ XGBoost
- ✅ LightGBM

### 5. Ensemble
Combined top 3 models in a **Soft Voting Ensemble** with XGBoost weighted 2x due to superior individual performance.

---

## 📈 Results

### Model Comparison

| Model | Train AUC | Test AUC | Gap | Notes |
|---|---|---|---|---|
| Random Forest | 0.894 | 0.832 | 0.062 | Most overfitting |
| CatBoost | 0.896 | 0.833 | 0.063 | Most overfitting |
| LightGBM | 0.880 | 0.833 | 0.046 | Good generalization |
| **XGBoost** ⭐ | **0.867** | **0.835** | **0.032** | **Best generalization** |
| Ensemble | 0.879 | 0.835 | 0.045 | No gain over XGBoost |

### Final Model — XGBoost Classification Report

```
              precision    recall  f1-score   support
           0       0.89      0.75      0.81      1033
           1       0.52      0.74      0.61       374

    accuracy                           0.75      1407
   macro avg       0.70      0.74      0.71      1407
weighted avg       0.79      0.75      0.76      1407
```

> **Why XGBoost over the Ensemble?**
> The ensemble achieved identical AUC (0.835) with a larger train-test gap (0.045 vs 0.032).
> Since all models are tree-based, they make similar errors — ensembling provided no benefit.
> XGBoost is simpler, faster, and generalizes better.

---

## 💡 Key Business Insights

| # | Insight | Recommendation |
|---|---|---|
| 1 | **Month-to-month customers churn at ~42%** vs 3% for 2-year contracts | Offer discounts to move customers to longer contracts |
| 2 | **Most churn happens in first 12 months** | Focus retention efforts on new customers immediately |
| 3 | **Higher monthly charges = more churn** | Introduce loyalty pricing for high-value customers |
| 4 | **No OnlineSecurity/TechSupport = higher churn** | Bundle these services free for at-risk customers |
| 5 | **Electronic check users churn the most** | Incentivize switching to auto-pay methods |
| 6 | **Fiber optic customers churn more than DSL** | Investigate service quality issues with fiber optic |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```python
import joblib
import pandas as pd

# Load the best model
model = joblib.load('Models/best_xgb_model.pkl')

# Load new customer data
new_data = pd.read_csv('Data/new_customers.csv')

# Predict churn probability
churn_proba = model.predict_proba(new_data)[:, 1]
print(f"Churn probability: {churn_proba}")
```

---

## 💡 Key Learnings

- **Data quality matters more than model complexity** — the initial synthetic dataset produced 0.48 AUC regardless of algorithm; switching to real data immediately fixed everything
- **Pipelines prevent data leakage** — fitting the preprocessor inside the pipeline ensures test data never influences training
- **Lower train AUC isn't always bad** — XGBoost's lower train AUC (0.867 vs 0.896) indicated better generalization, not worse performance
- **Ensembling has limits** — combining models only helps when they make different mistakes; tree-based models tend to agree, so the ensemble added no value here
- **Business context beats accuracy** — recall of 0.74 on churners matters more than overall accuracy; missing a churner is costlier than a false alarm

---

## 🛠️ Tech Stack

`Python 3.11` `Scikit-Learn` `XGBoost` `LightGBM` `CatBoost` `Pandas` `NumPy` `Matplotlib` `Seaborn` `ydata-profiling` `Joblib` `SciPy`

---

## 👤 Author

**Saheri Adak**
- 🎓 Second Year Data Science Student
- GitHub: [@Saher-Adak](https://github.com/Saheri-Adak)
- LinkedIn: [Saheri Adak]([(https://www.linkedin.com/in/saheri-adak-99290030a])

---

⭐ *If you found this project helpful, please give it a star!*
