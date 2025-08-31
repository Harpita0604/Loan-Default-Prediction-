# loan_default_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib # Import joblib to save/load scaler and model
import os # Import os for path handling

# Define the path to your dataset and model/scaler files
DATA_PATH = "/content/loan default dataset.zip"
MODEL_PATH = "loan_default_model.pkl"
SCALER_PATH = "loan_default_scaler.pkl"

# --- Load and preprocess data and train model (if not already saved) ---
# This function will only run if the model and scaler files don't exist
def load_data_and_train_model():
    data = pd.read_csv(DATA_PATH)

    # Handle missing values
    for col in ['LoanAmount', 'LoanTerm', 'CreditScore']:
        if col in data.columns:
             if data[col].dtype in ['int64', 'float64']:
                 data[col] = data[col].fillna(data[col].mean())
             else:
                 # Handle non-numeric missing values if necessary (e.g., fill with mode or a placeholder)
                 data[col] = data[col].fillna(data[col].mode()[0] if not data[col].empty else 'Missing')


    # One-hot encode categorical columns (Ensure all expected columns are handled)
    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    categorical_cols_exist = [col for col in categorical_cols if col in data.columns]
    data = pd.get_dummies(data, columns=categorical_cols_exist, drop_first=True)


    # Drop LoanID column if it exists
    if 'LoanID' in data.columns:
        data = data.drop('LoanID', axis=1)

    # Separate features and target
    if 'Default' in data.columns:
        X = data.drop("Default", axis=1)
        y = data["Default"]
    else:
        st.error("Target column 'Default' not found in the dataset.")
        return None, None, None # Return None if target not found


    # Train-test split (needed for fitting scaler and model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling - Fit on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model - Fit on the scaled training data
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Save the trained model and fitted scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return scaler, model, X.columns.tolist()

# --- Load the trained model and scaler (or train if files don't exist) ---
@st.cache_resource # Cache the resource loading
def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        st.write("Loading pre-trained model and scaler...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # To get feature names, we still need to load data once
        data = pd.read_csv(DATA_PATH)
        categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
        categorical_cols_exist = [col for col in categorical_cols if col in data.columns]
        data = pd.get_dummies(data, columns=categorical_cols_exist, drop_first=True)
        if 'LoanID' in data.columns:
             data = data.drop('LoanID', axis=1)
        if 'Default' in data.columns:
            X = data.drop("Default", axis=1)
            feature_names = X.columns.tolist()
        else:
            feature_names = [] # Handle case where Default is missing

        return scaler, model, feature_names
    else:
        st.write("Training model and fitting scaler...")
        return load_data_and_train_model()


# Load scaler, model, and feature names using the caching function
scaler, model, feature_names = load_model_and_scaler()


# Check if loading/training was successful
if scaler is not None and model is not None and feature_names is not None:

    # ---------------- STREAMLIT APP ---------------- #
    st.title("💰 Loan Default Prediction App")
    st.write("Enter customer details below to check if they might default on a loan:")

    # Create a dictionary to store user inputs
    user_inputs = {}

    # Define input widgets for each feature based on expected type and range
    st.header("Customer Information")
    user_inputs['Age'] = st.slider("Age", 18, 90, 30)
    user_inputs['Income'] = st.number_input("Annual Income", min_value=0, max_value=1000000, value=50000)
    user_inputs['LoanAmount'] = st.number_input("Loan Amount", min_value=500, max_value=500000, value=10000)
    user_inputs['CreditScore'] = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    user_inputs['MonthsEmployed'] = st.number_input("Months Employed", min_value=0, max_value=300, value=12)
    user_inputs['NumCreditLines'] = st.number_input("Number of Credit Lines", min_value=0, max_value=20, value=2)
    user_inputs['InterestRate'] = st.number_input("Interest Rate", min_value=0.0, max_value=30.0, value=5.0, format="%.2f")
    user_inputs['LoanTerm'] = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
    user_inputs['DTIRatio'] = st.slider("DTI Ratio", 0.0, 1.0, 0.5, format="%.2f")

    st.header("Categorical Information")

    # Education (One-hot encoded: Education_High School, Education_Master's, Education_PhD)
    education_options = ["Bachelor's", "High School", "Master's", "PhD"]
    education_selected = st.selectbox("Education", education_options)
    for option in education_options:
        if option != "Bachelor's": # 'Bachelor\'s' is the base case dropped by drop_first=True
             user_inputs[f'Education_{option}'] = 1 if education_selected == option else 0


    # EmploymentType (One-hot encoded: EmploymentType_Part-time, EmploymentType_Self-employed, EmploymentType_Unemployed)
    employment_options = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    employment_selected = st.selectbox("Employment Type", employment_options)
    for option in ["Part-time", "Self-employed", "Unemployed"]:
         user_inputs[f'EmploymentType_{option}'] = 1 if employment_selected == option else 0


    # MaritalStatus (One-hot encoded: MaritalStatus_Married, MaritalStatus_Single)
    marital_options = ["Married", "Divorced", "Single"]
    marital_selected = st.selectbox("Marital Status", marital_options)
    for option in ["Married", "Single"]:
        user_inputs[f'MaritalStatus_{option}'] = 1 if marital_selected == option else 0
    # Handle 'Divorced' which is the base case
    if marital_selected == "Divorced":
         if 'MaritalStatus_Married' in user_inputs: user_inputs['MaritalStatus_Married'] = 0
         if 'MaritalStatus_Single' in user_inputs: user_inputs['MaritalStatus_Single'] = 0


    # HasMortgage (One-hot encoded: HasMortgage_Yes)
    user_inputs['HasMortgage_Yes'] = st.checkbox("Has Mortgage?")


    # HasDependents (One-hot encoded: HasDependents_Yes)
    user_inputs['HasDependents_Yes'] = st.checkbox("Has Dependents?")


    # LoanPurpose (One-hot encoded: LoanPurpose_Business, LoanPurpose_Education, LoanPurpose_Home, LoanPurpose_Other)
    purpose_options = ["Other", "Business", "Education", "Home", "Auto", "Debt Consolidation", "Home Improvement"] # Include all possible from dataset if known
    purpose_selected = st.selectbox("Loan Purpose", purpose_options)
    for option in ["Business", "Education", "Home", "Other"]: # Only include columns created by one-hot encoding
         user_inputs[f'LoanPurpose_{option}'] = 1 if purpose_selected == option else 0


    # HasCoSigner (One-hot encoded: HasCoSigner_Yes)
    user_inputs['HasCoSigner_Yes'] = st.checkbox("Has CoSigner?")


    # --- Ensure all 24 features are in user_inputs, filling missing one-hot encoded ones with 0 ---
    # Create a dictionary with all feature names initialized to 0
    user_data_dict = {feature: 0 for feature in feature_names}
    # Update with actual user inputs
    for key, value in user_inputs.items():
         if key in user_data_dict:
             # Convert boolean checkboxes to int (True=1, False=0)
             if isinstance(value, bool):
                 user_data_dict[key] = int(value)
             else:
                 user_data_dict[key] = value
         # Handle the case where a one-hot encoded column was generated
         elif key in feature_names:
             user_data_dict[key] = value # This covers the 1 or 0 set by selectbox logic


    # --- Prediction Logic ---
    if st.button("Predict Loan Default"):
        # Create a DataFrame from user inputs, ensuring correct column order
        user_data_df = pd.DataFrame([user_data_dict], columns=feature_names)

        # Scale user input
        user_data_scaled = scaler.transform(user_data_df)
        user_data_scaled_df = pd.DataFrame(user_data_scaled, columns=feature_names) # Convert back to DataFrame with names

        # Make prediction
        prediction = model.predict(user_data_scaled_df)
        prediction_proba = model.predict_proba(user_data_scaled_df)[:, 1] # Get probability of default

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"⚠️ Prediction: **Likely to Default** (Probability: {prediction_proba[0]:.2f})")
        else:
            st.success(f"✅ Prediction: **Not Likely to Default** (Probability: {prediction_proba[0]:.2f})")

else:
    st.error("Failed to load model and scaler. Please check the data path and training process.")
