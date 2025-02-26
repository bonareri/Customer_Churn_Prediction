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
#### Summary Statistics

<img width="380" alt="EDA" src="https://github.com/user-attachments/assets/6c9877be-8c09-471f-a0a8-b57f000c0fd4" />

**Key Insights**  
- Most customers are **not senior citizens**.  
- Many customers are **new (tenure = 0)**.  
- **Wide variation in MonthlyCharges**, suggesting different service levels.  
- Some customers have **TotalCharges = 0**, likely due to recent sign-ups.

#### Churn Distribution

![image](https://github.com/user-attachments/assets/1b42599f-deef-4ff1-be13-24795f24b410)

  **Key Insights:**
- The dataset has an **imbalanced churn distribution**, with significantly more customers who did not churn.
- This imbalance may impact model training, requiring techniques like **resampling** or adjusting class weights.
- A **26.5% churn rate** suggests that a considerable portion of customers are leaving, indicating potential business challenges.

#### Distribution of Numerical Features

![image](https://github.com/user-attachments/assets/229d4ef8-d510-4bb0-8da5-f77603e88857)

  **Key Insights**  
- **Tenure is right-skewed**, with many short-term and long-term customers.  
- **Monthly Charges show a bimodal distribution**, suggesting tiered pricing plans.  
- **Total Charges is highly right-skewed**, with newer customers having lower totals.  
- **Mean > Median for all features**, indicating the influence of high-value customers.  
- **Churn is likely linked to pricing and tenure**, as short-term customers tend to leave early.  

#### Feature Correlation

![image](https://github.com/user-attachments/assets/fa85616e-e0d7-4668-a799-f960d8defe8b)

- **Key Insights**
- Customers with **fiber optic internet, electronic check payments, and high monthly charges** are more likely to churn.
- Long-term contracts (**two-year contracts**), **higher tenure**, and **added security/support services** reduce churn.
- **Online security, tech support, and total charges** negatively correlate with churn, suggesting customers using these services tend to stay.

These insights guide **feature selection** for predictive modeling and targeted customer retention strategies.

## Machine Learning Models

### Random Forest  

Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and reduce overfitting. It operates based on the principle of **bagging (bootstrap aggregation)**, where multiple trees are trained on different subsets of data.  

**How It Works**  
1. **Bootstrapping**: The model creates multiple random subsets of the training data with replacement.  
2. **Decision Trees**: Each subset is used to train an independent decision tree.  
3. **Random Feature Selection**: At each split, only a random subset of features is considered, ensuring diversity among trees.  
4. **Aggregation**:  
   - For classification, the majority vote across trees determines the final prediction.  
   - For regression, the average of all tree predictions is taken.  

![image](https://github.com/user-attachments/assets/f7f6547d-f40c-45cb-871f-f7944dfd560b)

**Advantages**  
✅ **Reduces Overfitting**: Combining multiple trees prevents individual trees from overfitting to training data.  
✅ **Handles Missing Data & Outliers**: Can work well with noisy data.  
✅ **Feature Importance**: Provides insights into which features contribute most to predictions.  
✅ **Works Well with Non-Linear Data**: Suitable for complex decision boundaries.  

### XGBoost (Extreme Gradient Boosting) 

XGBoost is a powerful gradient boosting algorithm optimized for speed and performance. It builds trees sequentially, learning from previous errors to enhance predictions.  

**How It Works**  
1. **Initialize Predictions**: The model starts with a base prediction, typically the mean (for regression) or a probability score (for classification).  
2. **Compute Residuals (Errors)**: The difference between actual and predicted values is calculated to identify areas where the model needs improvement.  
3. **Train Weak Learners (Decision Trees)**: A small decision tree is trained to predict the residuals rather than the actual target values.  
4. **Gradient Descent Update**: Instead of minimizing errors directly, XGBoost updates predictions using gradient descent to optimize performance.  
5. **Apply Regularization**: L1 (Lasso) and L2 (Ridge) regularization prevent overfitting, and shrinkage (learning rate) ensures stability.  
6. **Repeat Until Convergence**: The process is repeated multiple times, adding new trees iteratively to correct errors.  
7. **Make Final Predictions**: The final prediction is obtained by combining all weak learners using weighted averaging (for regression) or majority voting (for classification).

![image](https://github.com/user-attachments/assets/3e9b6ebe-64b0-4c31-bc17-cb4744869e52)

**Why XGBoost is Powerful?**  
✅ Handles missing values automatically  
✅ Regularization (L1 & L2) reduces overfitting  
✅ Parallelized execution speeds up training  
✅ Optimized for large datasets  
✅ Supports early stopping for efficiency  

### **Support Vector Machine (SVM) with RBF Kernel**  

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. The **Radial Basis Function (RBF) kernel**, also known as the Gaussian kernel, is one of the most commonly used kernels in SVM due to its ability to handle non-linearly separable data.  

**How It Works**  
1. **Mapping to a Higher-Dimensional Space**  
   - When data is not linearly separable in its original space, the **RBF kernel** transforms it into a higher-dimensional space where a linear decision boundary can be found.  
   - This transformation is done implicitly using a kernel trick, avoiding the need to compute explicit feature mappings.  

2. **Measuring Similarity**  
   - The RBF kernel computes the similarity between two data points **x** and **x'** using the formula:  
     \[
     K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)
     \]  
   - Here, **γ (gamma)** controls the influence of a single training example.  
     - **Higher γ** → Closer data points have more influence.  
     - **Lower γ** → Data points farther apart can still affect each other.  

3. **Finding the Optimal Hyperplane**  
   - The SVM algorithm finds a **maximum-margin hyperplane** in the transformed space.  
   - Support vectors (critical data points) help define the decision boundary.  

4. **Regularization with C Parameter**  
   - The **C parameter** controls the trade-off between maximizing the margin and minimizing classification errors.  
     - **High C** → Low bias, high variance (tighter margin).  
     - **Low C** → High bias, low variance (wider margin).  

**Pros**:  
- Handles non-linearly separable data well.  
- Effective in high-dimensional spaces.  
- Less affected by outliers.  

## Model Evaluation
### Metrics Used

- **Accuracy**: Measures the percentage of correctly classified instances out of the total. Best for balanced datasets.  
- **Precision & Recall**:  
  - **Precision**: The proportion of correctly predicted positive cases out of all predicted positives (useful when false positives are costly).  
  - **Recall**: The proportion of actual positive cases correctly identified (important when missing positives is costly).  
- **F1-score**: The harmonic mean of precision and recall, balancing both in cases of imbalanced datasets.  
- **ROC-AUC Score**: Measures how well the model distinguishes between classes, with higher values indicating better performance across different thresholds.  

 - **Hyperparameter Tuning**: Used **RandomizedSearchCV** to optimize model parameters.

### 📌 Random Forest Performance Before Tuning:

- Accuracy: 0.76
- Precision: 0.88 (Class 0), 0.54 (Class 1)
- Recall: 0.79 (Class 0), 0.70 (Class 1)
- F1-score: 0.83 (Class 0), 0.61 (Class 1)

![image](https://github.com/user-attachments/assets/2943cf20-b6bb-4e7c-986f-2eb8aa9988f0)

- The model performs well for predicting "No Churn" cases (Class 0) but struggles with predicting actual churn cases (Class 1).

**Ovefitting Analysis**

![image](https://github.com/user-attachments/assets/a705cf9f-1b5a-47ba-9680-7bb39ccb7a02)

### 📌 Random Forest Performance After Tuning:

- Accuracy: 0.76
- Precision: 0.87 (Class 0), 0.54 (Class 1)
- Recall: 0.79 (Class 0), 0.67 (Class 1)
- F1-score: 0.83 (Class 0), 0.60 (Class 1)

![image](https://github.com/user-attachments/assets/da7a4e54-fa7d-47b0-b002-db14041d020e)

### 📌 SVM Performance:

- Accuracy: 0.78
- Precision: 0.86 (Class 0), 0.57 (Class 1)
- Recall: 0.83 (Class 0), 0.64 (Class 1)
- F1-score: 0.84 (Class 0), 0.60 (Class 1)

![image](https://github.com/user-attachments/assets/0651ee67-6f30-424f-b860-9f41af97cb92)

### 📌 XGBoost Performance Before Tuning:

- Accuracy: 0.76
- Precision: 0.88 (Class 0), 0.54 (Class 1)
- Recall: 0.78 (Class 0), 0.71 (Class 1)
- F1-score: 0.83 (Class 0), 0.61 (Class 1)

![image](https://github.com/user-attachments/assets/0081e53a-7f24-46ef-a9b1-1794bc40cec5)

### 📌 XGBoost Performance After Tuning:

- Accuracy: 0.75
- Precision: 0.85 (Class 0), 0.53 (Class 1)
- Recall: 0.81 (Class 0), 0.60 (Class 1)
- F1-score: 0.83 (Class 0), 0.56 (Class 1)

![image](https://github.com/user-attachments/assets/447a1412-1aba-45ee-bc1e-34039774998f)

### Model Performance Summary

| Model                     | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
|---------------------------|----------|---------------------|------------------|------------------  |
| **Random Forest (Tuned)**  | **0.7600**  | **0.54**             | **0.67**         | **0.60**          |
| **Random Forest (Baseline)** | **0.7600**  | **0.54**             | **0.70**         | **0.61**          |
| **SVM**                   | **0.7771**  | **0.57**             | **0.64**         | **0.60**          |
| **XGBoost**               | **0.7601**  | **0.54**             | **0.71**         | **0.61**          |
| **XGBoost (Tuned)**       | **0.7530**  | **0.53**             | **0.60**         | **0.56**          |

### **Best Model**
- Since telecom churn prediction focuses on identifying at-risk customers to prevent churn, recall is crucial.
- Missing a churner (false negative) means losing a customer, which is costly.
- XGBoost (Baseline) is the best choice based on recall (0.71), meaning it captures the most churners.

## Conclusion

### Insights
- The analysis of telecom customer churn reveals key factors influencing customer retention, such as **contract type, monthly charges, tenure, and service subscriptions**.
- Customers with **month-to-month contracts** exhibit a higher likelihood of churning compared to those with longer-term contracts.
- **High monthly charges** correlate with higher churn rates, indicating that pricing plays a crucial role in customer retention.
- Customers who do not subscribe to additional services like **tech support and streaming services** tend to leave at a higher rate.
- **Longer tenure** customers are less likely to churn, suggesting that loyalty programs may play a role in reducing churn.

### Recommendations
1. **Introduce Loyalty Programs**
   - Provide incentives for long-term contracts, such as discounts or additional services, to reduce churn among month-to-month subscribers.

2. **Optimize Pricing Strategies**
   - Offer tiered pricing or bundled services to cater to different customer segments and reduce dissatisfaction with high monthly charges.

3. **Enhance Customer Support & Service Offerings**
   - Encourage the adoption of additional services, such as tech support and premium subscriptions, which correlate with lower churn rates.
   - Implement proactive customer support to identify and address customer concerns before they result in churn.

4. **Personalized Retention Strategies**
   - Utilize predictive modeling to identify at-risk customers and engage them with targeted offers, personalized discounts, or exclusive features.

5. **Improve Customer Engagement**
   - Conduct surveys and gather feedback from customers to understand pain points and improve service delivery.
   - Increase engagement through mobile apps, SMS notifications, or loyalty rewards to keep customers actively involved with the brand.

By implementing these strategies, telecom companies can proactively reduce churn, improve customer satisfaction, and enhance overall business profitability.
