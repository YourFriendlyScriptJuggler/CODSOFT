# Machine Learning Projects - CodSoft Internship

This repository contains three machine learning projects developed during my CodSoft internship. Each project addresses a real-world problem using various machine learning algorithms and data preprocessing techniques.

---

## 1. SMS Spam Detection

### Project Goal
To build an AI model capable of classifying SMS messages as either "spam" or "legitimate" (ham).

### Dataset
The project utilizes a publicly available dataset of SMS messages, labeled as 'ham' or 'spam'.

### Key Techniques & Algorithms
* **Data Preprocessing**:
    * Text cleaning: Lowercasing, punctuation removal, stop word removal.
    * Stemming (using PorterStemmer) for word normalization.
* **Feature Extraction**:
    * TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
* **Machine Learning Models**:
    * **Multinomial Naive Bayes**: A probabilistic classifier suitable for text classification.
    * **Logistic Regression**: A linear model for binary classification.
    * **Support Vector Machine (SVC)**: A powerful algorithm for classification tasks.

### Performance Highlights
The models were evaluated using `accuracy_score`, `precision_score`, and `confusion_matrix`. The project demonstrates effective identification of spam messages.

### Files
* `SmsSpamDetection.ipynb`: Jupyter Notebook containing the complete code for data preprocessing, model training, and evaluation.

---

## 2. Customer Churn Prediction

### Project Goal
To develop a machine learning model that predicts customer churn for a subscription-based service or business, enabling proactive retention strategies.

### Dataset
The project uses the 'Churn Modelling' dataset sourced from Kaggle, which contains 10,000 rows of bank customer data.

### Key Features in Dataset
* **Demographics**: Gender, Age, Geography.
* **Account Details**: Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary.
* **Target Variable**: 'Exited' (indicating churn).

### Key Techniques & Algorithms
* **Data Preprocessing**:
    * Handling categorical features (e.g., 'Gender', 'Geography') using Label Encoding and One-Hot Encoding.
    * Numerical feature scaling using `StandardScaler`.
* **Exploratory Data Analysis (EDA)**:
    * Visualizations to understand churn distribution across various customer attributes.
* **Machine Learning Models**:
    * **Logistic Regression**: A baseline linear model.
    * **Random Forest Classifier**: An ensemble learning method providing high accuracy.
    * **XGBoost Classifier**: A highly efficient and flexible gradient boosting implementation.

### Performance Highlights
Models were evaluated using accuracy, precision, recall, and confusion matrices. The **Random Forest Classifier** emerged as the top performer, achieving an impressive **86.55% accuracy** on unseen data.

### Files
* `Customer_Churn_Prediction_ML.ipynb`: Jupyter Notebook with comprehensive code for data loading, preprocessing, EDA, model training, and evaluation.
* `Churn_Modelling.csv`: The dataset used for this project.

---

## 3. Credit Card Fraud Detection

### Project Goal
To build a robust AI model that can accurately detect fraudulent credit card transactions, minimizing financial losses for consumers and businesses.

### Dataset
The project utilizes a simulated credit card transaction dataset from Kaggle, encompassing transactions from 2019 to 2020. This dataset is characterized by extreme class imbalance (very few fraudulent transactions).

### Key Features in Dataset
* Transaction details, merchant information, customer demographics, and a 'is_fraud' indicator.

### Key Techniques & Algorithms
* **Data Preprocessing**:
    * Handling date and time features, extracting new time-based features (e.g., transaction hour, day).
    * Categorical variable encoding, potentially using Weight of Evidence (WOE) for imbalanced data.
* **Handling Imbalance**:
    * Down-sampling the majority class to create a more balanced dataset for effective model training.
* **Feature Scaling**:
    * `StandardScaler` applied to numerical features.
* **Machine Learning Models**:
    * **Logistic Regression**
    * **Decision Tree Classifier**
    * **Random Forest Classifier**
    * **XGBoost Classifier**

### Performance Highlights
Evaluation focused on Precision, Recall, and F1-score due to class imbalance. The **Random Forest Classifier** demonstrated exceptional performance, achieving a **precision of 99%** and a **recall of 98%**, proving its strong ability to identify fraudulent transactions.

### Files
* `fraud_detection_using_ml (1).ipynb`: Jupyter Notebook containing the full implementation.

 

## Contact

For any questions or collaborations, feel free to reach out.

---
