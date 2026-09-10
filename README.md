# Customer Churn Prediction
Project Overview

Customer churn refers to customers leaving or stopping their relationship with a business. The goal of this project is to build a machine learning model that can identify customers who are likely to churn.

This project uses customer demographic and account-related information to predict whether a customer is likely to Stay or Churn.

Objective

The main objectives of this project are:

Clean and prepare customer data
Handle missing and invalid values
Perform Exploratory Data Analysis (EDA)
Preprocess categorical features
Train different machine learning classification models
Select the most suitable model for churn prediction
Predict whether a customer is likely to churn
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
Joblib
Project Workflow

Raw Data → Data Cleaning → Exploratory Data Analysis → Data Preprocessing → Train-Test Split → Model Training → Model Evaluation → Balanced Random Forest → Churn Prediction

Data Cleaning

The dataset was cleaned by:

Handling missing values
Removing duplicate records
Correcting invalid Credit Score values
Correcting invalid Age values
Correcting invalid Tenure values
Handling negative Balance values
Correcting invalid Number of Products
Handling invalid Estimated Salary values
Standardizing categorical values
Exploratory Data Analysis

Some important findings from the analysis:

The dataset contains more customers who stayed than customers who churned.
Older customers showed a higher churn rate.
Germany had a higher churn rate compared with France and Spain.
Inactive members had a higher churn rate than active members.
Customers with different numbers of products showed significantly different churn rates.
Churned customers had a higher average balance than customers who stayed.
Machine Learning Models

Three classification models were evaluated:

Model	Accuracy	Churn Recall	Churn F1
Logistic Regression	81.29%	19%	29%
Random Forest	85.03%	39%	52%
Balanced Random Forest	82.98%	53%	56%
Final Model

The Balanced Random Forest model was selected as the final model.

Although the normal Random Forest achieved higher overall accuracy, Balanced Random Forest detected more customers who actually churned.

Since the main objective of this project is to identify customers who are likely to leave, churn recall and F1-score were considered more important than overall accuracy.
