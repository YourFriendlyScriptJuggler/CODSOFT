# ML-Projects-Showcase

This repository showcases a collection of machine learning projects I've developed, demonstrating my skills in various aspects of the ML pipeline, from data preprocessing and exploratory analysis to model building and evaluation. These projects were completed as part of my practical learning and internship experience.

---

## 1. SMS Spam Detection

### Project Goal
To build an AI model capable of classifying SMS messages as either "spam" or "legitimate" (ham). This project aims to enhance digital communication security by accurately filtering unwanted messages.

### Data Source
The dataset used for this project is a publicly available collection of SMS messages, commonly found on platforms like Kaggle, specifically designed for spam classification tasks.
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### Key Techniques & Algorithms
* **Data Preprocessing**:
    * Text cleaning: Lowercasing, punctuation removal, stop word removal.
    * Stemming (using PorterStemmer) for word normalization.
* **Feature Extraction**:
    * TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features, representing word importance.
* **Machine Learning Models**:
    * **Multinomial Naive Bayes**
    * **Logistic Regression**
    * **Support Vector Machine (SVC)**

### Performance Highlights
The models were rigorously evaluated using `accuracy_score`, `precision_score`, and `confusion_matrix`. The project demonstrates effective identification of spam messages with high precision and accuracy, ensuring minimal false positives for legitimate messages.

### Files
* `SmsSpamDetection.ipynb`: Jupyter Notebook containing the complete code for data preprocessing, model training, and evaluation.

**LinkedIn Post Link:** https://www.linkedin.com/posts/sulagna-dutta-ab5257358_machinelearning-spamdetection-nlp-activity-7353119667839750146-qFJK?utm_source=share&utm_medium=member_android&rcm=ACoAAFkGomkBXXLKhyNeNzm1_vfkud0iGjwlxtI
---

## 2. Customer Churn Prediction

### Project Goal
To develop a machine learning model that predicts customer churn for a subscription-based service or business. The objective is to identify customers at risk of leaving proactively, enabling targeted retention strategies.

### Data Source
The dataset used is the 'Churn Modelling' dataset, widely available on Kaggle. It provides comprehensive information about bank customers and their churn status.
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

### Key Features in Dataset
* **Demographics**: Gender, Age, Geography.
* **Account Details**: Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary.
* **Target Variable**: 'Exited' (1 if the customer churned, 0 otherwise).

### Key Techniques & Algorithms
* **Data Preprocessing**:
    * Handling categorical features (e.g., 'Gender', 'Geography') using Label Encoding and One-Hot Encoding.
    * Numerical feature scaling using `StandardScaler`.
* **Exploratory Data Analysis (EDA)**:
    * Visualizations to understand churn distribution and influencing factors across various customer attributes.
* **Machine Learning Models**:
    * **Logistic Regression**
    * **Random Forest Classifier**
    * **XGBoost Classifier**

### Performance Highlights
Models were evaluated using accuracy, precision, recall, and confusion matrices. The **Random Forest Classifier** emerged as the top performer, achieving an impressive **86.55% accuracy** on unseen data, demonstrating strong predictive capability for customer retention.

### Files
* `Customer_Churn_Prediction_ML.ipynb`: Jupyter Notebook with comprehensive code for data loading, preprocessing, EDA, model training, and evaluation.
* `Churn_Modelling.csv`: The raw dataset used for this project.

**LinkedIn Post Link:** https://www.linkedin.com/posts/sulagna-dutta-ab5257358_machinelearning-churnprediction-datascience-activity-7353111541883289600-Dx9G?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAFkGomkBXXLKhyNeNzm1_vfkud0iGjwlxtI&utm_campaign=copy_link
---

## 3. Credit Card Fraud Detection

### Project Goal
To build a robust AI model capable of accurately detecting fraudulent credit card transactions, thereby minimizing financial losses for consumers and businesses.

### Data Source
This project utilizes a simulated credit card transaction dataset sourced from Kaggle, encompassing transactions from 2019 to 2020. A key characteristic of this dataset is its extreme class imbalance, reflecting real-world fraud scenarios.
https://www.kaggle.com/datasets/kartik2112/fraud-detection

### Key Features in Dataset
* Transaction details, merchant information, customer demographics, and a 'is_fraud' indicator.

### Key Techniques & Algorithms
* **Data Preprocessing**:
    * Handling date and time features, extracting new time-based features (e.g., transaction hour, day).
    * Categorical variable encoding, potentially using Weight of Evidence (WOE) for imbalanced data.
* **Handling Imbalance**:
    * Down-sampling the majority class to create a more balanced dataset, crucial for effective training on rare fraud cases.
* **Feature Scaling**:
    * `StandardScaler` applied to numerical features.
* **Machine Learning Models**:
    * **Logistic Regression**
    * **Decision Tree Classifier**
    * **Random Forest Classifier**
    * **XGBoost Classifier**

### Performance Highlights
Evaluation focused on Precision, Recall, and F1-score due to the critical nature of identifying fraud and the dataset's imbalance. The **Random Forest Classifier** demonstrated exceptional performance, achieving a **precision of 99%** and a **recall of 98%** on the test set, proving its strong ability to correctly identify fraudulent transactions while minimizing false positives.

### Files
* `fraud_detection_using_ml (1).ipynb`: Jupyter Notebook containing the full implementation.

 **LinkedIn Post Link:** https://www.linkedin.com/posts/sulagna-dutta-ab5257358_machinelearning-frauddetection-creditcardfraud-activity-7353118020820504577-PCe_?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAFkGomkBXXLKhyNeNzm1_vfkud0iGjwlxtI&utm_campaign=copy_link

## Connect with Me

Feel free to connect with me on LinkedIn to discuss these projects or other machine learning topics!



---
