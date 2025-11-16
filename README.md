
# 💸 EMI Eligibility & EMI Amount Prediction

This project predicts:

✅ Whether a user is eligible for an EMI (loan) \
✅ How much EMI amount they can safely afford

It uses a Machine Learning classification model to determine eligibility and a regression model to estimate the maximum EMI amount.
The frontend is built using Streamlit, and the whole workflow is fully automated — from preprocessing to prediction.

## 🚀 Features 

### 🔍1. EMI Eligibility Prediction

    Predicts whether the applicant is:

        Eligible
        High-Risk
        Not Eligible

### 💰 2. EMI Amount Prediction

    If the applicant is eligible, the model predicts the maximum monthly EMI amount they can safely pay.

### 🧠 3. ML Models Used

    i. Classification Models

        Logistic Regression (baseline)
        Random Forest Classifier
        XGBoost Classifier (best performer)

    ii. Regression Models

        Linear Regression
        Random Forest Regressor
        XGBoost Regressor (best performer)

### 📊 4. End-to-End ML Pipeline

    Raw dataset handling
    EDA + Outlier Detection
    Data cleaning & type correction
    Categorical encoding
    Feature engineering
    Model training, evaluation & selection
    Model saving using Joblib
    Streamlit App for deployment

### 🏗 5. Deployment Ready

    Easily deployable on Streamlit Cloud.
    Live at : https://emiprediciton.streamlit.app

## 📁 Project Structure

    📦 emi-prediction-project
    │
    ├── 📂 data
    │   └── emi_dataset.csv
    │
    ├── 📂 models
    │   ├── best_classifier_model.pkl
    │   └── best_regression_model.pkl
    │
    ├── 📂 notebooks
    │   └── emi_prediction.ipynb
    │
    ├── 📂 app
    │     └──app.py
    ├── requirements.txt
    └── README.md

## 🛠 Tech Stack

    1.Python
    2.Pandas, NumPy
    3.XGBoost, Scikit-learn
    4.Matplotlib, Seaborn
    5.Streamlit
    6.Joblib
    7.GitHub for version control

## 📈 Model Performance
### 🟦 Best Classification Model — XGBoost
    Metric	                Score
    Accuracy	            0.988
    Precision (Macro)	    0.958
    Recall (Macro)	        0.937
    F1 Score (Macro)	    0.947

### 🟨 Best Regression Model — XGBoost
    Metric	                Score
    MAE	                    565
    RMSE	                930
    R²	                    0.985

## 🖥 Run Locally

### 1️⃣ Clone the repo
    git clone https://github.com/SouravPaul2002/emi_prediciton.git
    cd emi_prediction
### 2️⃣ Install dependencies
    pip install -r requirements.txt
### 3️⃣ Start Streamlit
    streamlit run app.py

## 🎯 What You Will Learn From This Project

    1.Complete ML pipeline development
    2.Handling mixed-type data & categorical encoding
    3.Feature engineering for finance datasets
    4.Building both classification & regression models
    5.Saving & loading models efficiently
    6.Building front-end UI with Streamlit
    7.Deploying ML apps
