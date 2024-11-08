

# 📊 Customer Churn Prediction

This project aims to predict customer churn (i.e., whether a customer will leave a bank) using machine learning techniques. Churn prediction is crucial for businesses to retain customers, optimize marketing strategies, and improve overall customer satisfaction. This project utilizes a dataset sourced from Kaggle and covers the complete data science pipeline from data preprocessing to model evaluation.

## 🚀 Project Overview

- **Objective**: Predict whether a bank customer will churn based on various customer attributes.
- **Dataset**: The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers?select=Churn_Modelling.csv).
- **Tech Stack**: Python, Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.

---

## 📂 Dataset Information

The dataset consists of customer data with the following features:
- **CustomerID**: Unique identifier for each customer.
- **CreditScore**: Customer's credit score.
- **Geography**: Customer's country of residence.
- **Gender**: Male or Female.
- **Age**: Customer's age.
- **Tenure**: Number of years with the bank.
- **Balance**: Account balance.
- **NumOfProducts**: Number of bank products used by the customer.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: Estimated annual salary of the customer.
- **Exited**: The target variable (1 = Customer Churned, 0 = Customer Retained).

---

## ⚙️ Installation and Setup

To run this project on your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Customer_churn_prediction.ipynb
   ```

---

## 🔍 Exploratory Data Analysis (EDA)

- **Data Visualization**: Analyzed customer demographics, account activities, and churn rates using visualizations.
- **Correlation Analysis**: Examined correlations between features like credit score, age, balance, and churn.
- **Key Insights**:
  - Older customers have a higher churn rate.
  - Customers with a lower number of products are more likely to churn.
  - There is a noticeable difference in churn rates across different geographic regions.

---

## 🛠️ Data Preprocessing

- **Handling Missing Values**: Checked for missing data and handled appropriately.
- **Feature Encoding**: Converted categorical variables (`Geography`, `Gender`) using one-hot encoding.
- **Feature Scaling**: Applied feature scaling using StandardScaler for numerical columns to improve model performance.
- **Data Splitting**: Split the dataset into training and testing sets (80% train, 20% test).

---

## 🤖 Modeling and Evaluation

Several machine learning models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM)**
5. **XGBoost Classifier**

- **Model Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score

- **Best Performing Model**: The `Random Forest Classifier` achieved the highest accuracy with an F1-Score of 0.84 and an AUC-ROC of 0.89.

---

## 📈 Results and Insights

- **Key Findings**:
  - Active customers with a high balance and multiple bank products are less likely to churn.
  - The model identified customers at high risk of churn with an accuracy of 85%, allowing targeted retention strategies.
- **Business Impact**: Implementing this model could help the bank reduce churn rates by focusing on at-risk customers.

---

## 🧑‍💻 How to Use

To make predictions on new customer data:
1. Ensure the new data follows the same format as the original dataset.
2. Load the trained model (`model.pkl`) using Python's `pickle` library.
3. Use the `.predict()` method to classify whether a customer will churn.

Example:
```python
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict on new data
new_customer = [[600, 'France', 'Female', 40, 3, 60000, 2, 1, 1, 50000]]
prediction = model.predict(new_customer)
print("Churn" if prediction[0] == 1 else "Retained")
```

---

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/new-feature`)
3. Commit your Changes (`git commit -m 'Add new feature'`)
4. Push to the Branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

Thank you for reading!
