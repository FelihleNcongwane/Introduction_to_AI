import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the dataset
dataset_path = '/content/drive/My Drive/Colab Notebooks/Assignment3/CustomerChurn_dataset.csv'
df = pd.read_csv(dataset_path)

# Preprocessing functions
def preprocess_data(data):
    # Select the top 10 features
    selected_features = ['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract_Month-to-month', 'InternetService_Fiber optic',
                         'PaymentMethod_Electronic check', 'gender', 'PaperlessBilling', 'Partner', 'OnlineBackup']
    data = data[selected_features]

    # Convert binary columns to numeric
    yes_no_columns = ['Partner', 'PaperlessBilling']
    for col in yes_no_columns:
        data[col].replace({'Yes': 1, 'No': 0}, inplace=True)

    # Convert 'gender' column to numeric
    data['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

    # One-hot encode categorical columns
    categorical_columns = ['Contract_Month-to-month', 'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 'OnlineBackup']
    data = pd.get_dummies(data=data, columns=categorical_columns)

    # Scale numerical columns
    scaler = StandardScaler()
    numerical_columns = ['TotalCharges', 'MonthlyCharges', 'tenure']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data

# Load the model
model_path = '/content/drive/My Drive/Colab Notebooks/AI_Assignment3/my_model.h5'
model = tf.keras.models.load_model(model_path)

# Churn Prediction function
def predict_churn(data):
    preprocessed_data = preprocess_data(data)
    X = preprocessed_data.values
    y_pred = model.predict(X)
    return y_pred

# Streamlit app
def main():
    st.title("Churn Prediction App")

    # Get user input for new customer data
    st.header("Enter New Customer Data")
    total_charges = st.number_input("Total Charges")
    monthly_charges = st.number_input("Monthly Charges")
    tenure = st.number_input("Tenure")
    contract_monthly = st.selectbox("Contract (Month-to-month)", ['No', 'Yes'])
    internet_fiber_optic = st.selectbox("Internet Service (Fiber optic)", ['No', 'Yes'])
    payment_electronic = st.selectbox("Payment Method (Electronic check)", ['No', 'Yes'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    online_backup = st.selectbox("Online Backup", ['No', 'Yes'])

    if st.button("Predict Churn"):
        new_data = pd.DataFrame({
            'TotalCharges': [total_charges],
            'MonthlyCharges': [monthly_charges],
            'tenure': [tenure],
            'Contract_Month-to-month': [1 if contract_monthly == 'Yes' else 0],
            'InternetService_Fiber optic': [1 if internet_fiber_optic == 'Yes' else 0],
            'PaymentMethod_Electronic check': [1 if payment_electronic == 'Yes' else 0],
            'gender': [1 if gender == 'Female' else 0],
            'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
            'Partner': [1 if partner == 'Yes' else 0],
            'OnlineBackup': [1 if online_backup == 'Yes' else 0]
        })

        predictions = predict_churn(new_data)

        churn_probability = predictions[0][0]
        churn_percentage = churn_probability * 100
        if churn_percentage > 50:
            st.warning("The customer is likely to churn with a probability of {:.2f}%.".format(churn_percentage))
        else:
            st.success("The customer is likely to stay with a probability of {:.2f}%.".format(100 - churn_percentage))

# Run the app
if __name__ == "__main__":
    main()