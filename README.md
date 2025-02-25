# Telecom Customer Churn Prediction

## Overview
Customer churn is the percentage of customers who stop using a company's product or service during a certain timeframe. Customer churn is a major challenge in the telecom industry. This project aims to predict whether a customer will churn based on historical data, allowing telecom companies to implement proactive retention strategies.

## Dataset
The dataset contains **7,043 customer records** with **21 features**, including demographic details, account information, and service usage patterns. The target variable is **`Churn`**, which indicates whether a customer has left (`Yes`) or stayed (`No`).

### Features
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target Variable:** `Churn`

## Project Workflow

### 1. Data Preprocessing  
- **Handling Missing Values**: Missing values in **TotalCharges** were imputed using tenure-based methods. Specifically, entries with **tenure = 0** were set to **TotalCharges = 0** since they represent new customers who have not been billed yet.  
- **Encoding Categorical Variables**: Applied **Label Encoding** for binary categorical features and **One-Hot Encoding** for multi-category features to avoid multicollinearity.  
- **Feature Scaling**: Used **RobustScaler** for numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to handle skewed distributions and outliers effectively. 

### 2. Exploratory Data Analysis (EDA)
-**Summary Statistics**: 
<img width="380" alt="EDA" src="https://github.com/user-attachments/assets/6c9877be-8c09-471f-a0a8-b57f000c0fd4" />

**Key Insights**  
- Most customers are **not senior citizens**.  
- Many customers are **new (tenure = 0)**.  
- **Wide variation in MonthlyCharges**, suggesting different service levels.  
- Some customers have **TotalCharges = 0**, likely due to recent sign-ups.

- **Churn Distribution**:
  ![image](https://github.com/user-attachments/assets/1b42599f-deef-4ff1-be13-24795f24b410)

  **Key Insights:**
- The dataset has an **imbalanced churn distribution**, with significantly more customers who did not churn.
- This imbalance may impact model training, requiring techniques like **resampling** or adjusting class weights.
- A **26.5% churn rate** suggests that a considerable portion of customers are leaving, indicating potential business challenges.

- **Distribution of Numerical Features**:
  ![image](https://github.com/user-attachments/assets/229d4ef8-d510-4bb0-8da5-f77603e88857)

  **Key Insights**  
- **Tenure is right-skewed**, with many short-term and long-term customers.  
- **Monthly Charges show a bimodal distribution**, suggesting tiered pricing plans.  
- **Total Charges is highly right-skewed**, with newer customers having lower totals.  
- **Mean > Median for all features**, indicating the influence of high-value customers.  
- **Churn is likely linked to pricing and tenure**, as short-term customers tend to leave early.  

- **Feature Correlation**:
  ![image](https://github.com/user-attachments/assets/fa85616e-e0d7-4668-a799-f960d8defe8b)

- **Key Insights**
- Customers with **fiber optic internet, electronic check payments, and high monthly charges** are more likely to churn.
- Long-term contracts (**two-year contracts**), **higher tenure**, and **added security/support services** reduce churn.
- **Online security, tech support, and total charges** negatively correlate with churn, suggesting customers using these services tend to stay.

These insights guide **feature selection** for predictive modeling and targeted customer retention strategies.

## Machine Learning Models

### **Random Forest**  

Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and reduce overfitting. It operates based on the principle of **bagging (bootstrap aggregation)**, where multiple trees are trained on different subsets of data.  

**How It Works**  
1. **Bootstrapping**: The model creates multiple random subsets of the training data with replacement.  
2. **Decision Trees**: Each subset is used to train an independent decision tree.  
3. **Random Feature Selection**: At each split, only a random subset of features is considered, ensuring diversity among trees.  
4. **Aggregation**:  
   - For classification, the majority vote across trees determines the final prediction.  
   - For regression, the average of all tree predictions is taken.  

## 🌳 Random Forest Algorithm Flowchart

```mermaid
graph TD;
    A[Start] --> B{Bootstrap Sampling};
    B -->|Sample 1| C[Decision Tree 1];
    B -->|Sample 2| D[Decision Tree 2];
    B -->|Sample N| E[Decision Tree N];
    C --> F[Prediction 1];
    D --> G[Prediction 2];
    E --> H[Prediction N];
    F --> I{Majority Voting / Averaging};
    G --> I;
    H --> I;
    I --> J[Final Prediction];


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



