# Telecom Customer Churn Prediction

## Overview
Customer churn is a major challenge in the telecom industry. This project aims to predict whether a customer will churn based on historical data, allowing telecom companies to implement proactive retention strategies.

## Dataset
The dataset contains **7,043 customer records** with **21 features**, including demographic details, account information, and service usage patterns. The target variable is **`Churn`**, which indicates whether a customer has left (`Yes`) or stayed (`No`).

### Features
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target Variable:** `Churn`

## Project Workflow

### 1. Data Preprocessing
- **Handling Missing Values**: Missing values in `TotalCharges` were imputed using tenure-based methods.
- **Encoding Categorical Variables**: Used **One-Hot Encoding** and **Label Encoding**.
- **Feature Scaling**: Standardized numerical features (`MonthlyCharges`, `TotalCharges`) to improve model performance.

### 2. Exploratory Data Analysis (EDA)
- **Churn Distribution**: Visualized the proportion of customers who churned.
- **Feature Correlation**: Identified key variables influencing churn.
- **Feature Importance**: Used statistical methods to highlight impactful features.

### 3. Feature Engineering
- **Created Derived Features**: Segmented `tenure` into groups.
- **Removed Redundant Features**: Eliminated highly correlated variables to prevent multicollinearity.

## Machine Learning Models

### Logistic Regression
- **Concept**: A statistical model for binary classification that predicts the probability of churn.
- **Mathematics**: Uses the **sigmoid function** to map inputs to probabilities.
- **Pros**: Simple, interpretable, and effective for linear relationships.
- **Cons**: Assumes a linear relationship between independent variables and the log-odds.

### Decision Tree
- **Concept**: A tree-like model that splits the dataset based on the most significant features.
- **Splitting Criteria**: Uses **Gini impurity** or **Entropy**.
- **Pros**: Easy to interpret and works well with both numerical and categorical data.
- **Cons**: Prone to overfitting without pruning.

### Random Forest
- **Concept**: An ensemble of multiple decision trees.
- **Mechanism**: Uses **bagging (bootstrap aggregation)** to reduce variance and improve accuracy.
- **Pros**: Reduces overfitting, provides feature importance.
- **Cons**: Computationally expensive.

### XGBoost (Extreme Gradient Boosting)
- **Concept**: A gradient boosting algorithm optimized for speed and performance.
- **Mechanism**: Sequentially improves weak models using boosting techniques.
- **Pros**: High predictive power, handles missing values well.
- **Cons**: Requires careful hyperparameter tuning.

## Model Evaluation
- **Metrics Used**:
  - Accuracy
  - Precision & Recall
  - F1-score
  - ROC-AUC Score
- **Hyperparameter Tuning**: Used **GridSearchCV** and **RandomizedSearchCV** to optimize model parameters.


## Installation & Usage

### Prerequisites
- **Python 3.8+**
- **Jupyter Notebook / Google Colab**
- **Required Libraries**:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost



