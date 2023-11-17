# Introduction_to_AI
Certainly! Here's an updated version of the README file that includes information about the model training and feature selection process:

```
# Churn Prediction App

This is a simple web application that predicts customer churn using a pre-trained machine learning model. It allows users to input customer data and get predictions on whether a customer is likely to churn or stay.

## Getting Started

To use the Churn Prediction App, you need to follow these steps:

1. Install the required packages by running the following command:

   ```
   pip install streamlit
   ````

2. Clone the repository:

   ```
   git clone <repository_url>
   ````

3. Run the application using the following command:

   ```
   streamlit run app.py
   ````

## Model Training

The machine learning model used for churn prediction was trained using a combination of Random Forest Classifier and a neural network architecture built with TensorFlow and Keras.

The feature selection process involved the following steps:

1. A Random Forest Classifier was trained on the dataset using the `fit` method.
2. The feature importances were obtained from the trained random forest model using the `feature_importances_` attribute.
3. The feature importances were stored in a Pandas Series and sorted in descending order.
4. The top 10 features were selected based on their importance using slicing on the sorted feature importances.

Once the top 10 features were identified, they were used as input to the neural network model.

The complete code for model training and feature selection can be found in the `train_model.py` file.

## Usage

Once the application is running, you can access it by opening the provided URL in your browser. The app will display a form where you can enter the customer data for prediction.

The following input fields are available:

- Total Charges: The total charges incurred by the customer.
- Monthly Charges: The monthly charges paid by the customer.
- Tenure: The duration of the customer's relationship with the company.
- Contract (Month-to-month): Whether the customer has a month-to-month contract.
- Internet Service (Fiber optic): Whether the customer has fiber optic internet service.
- Payment Method (Electronic check): Whether the customer uses electronic check as the payment method.
- Gender: The gender of the customer.
- Paperless Billing: Whether the customer has opted for paperless billing.
- Partner: Whether the customer has a partner.
- Online Backup: Whether the customer has online backup service.

Click the "Predict Churn" button to get the prediction result. The app will display a message indicating whether the customer is likely to churn or stay, along with
