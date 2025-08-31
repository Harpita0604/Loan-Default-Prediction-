import sklearn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------
# Upload Dataset
# ------------------------------
st.title("💳 Loan Default Prediction App")

st.write("Upload your dataset and predict if a loan applicant will default or not.")

# Add a clear heading and instructions for uploading the dataset
st.subheader("⬆️ Upload Your Dataset")
st.write("Look for the file upload box below to select your CSV file.")
uploaded_file = st.file_uploader("📂 Choose a CSV file from your system", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write("Here's a preview of the data you uploaded:")
    st.write(data.head())

    # ------------------------------
    # Preprocessing
    # ------------------------------
    st.sidebar.header("🔧 Model Settings")
    target_col = st.sidebar.selectbox("Select Target Column (Default)", data.columns, index=len(data.columns)-1)

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Handle 'Yes'/'No' columns before encoding
    for col in ['HasMortgage', 'HasCoSigner']:
        if col in X.columns:
            X[col] = X[col].map({'Yes': 1, 'No': 0})

    # Encode categorical columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ------------------------------
    # Model Training
    # ------------------------------
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"✅ Model trained successfully! Accuracy: *{accuracy*100:.2f}%*")

    # ------------------------------
    # User Input
    # ------------------------------
    st.sidebar.header("📝 Enter Applicant Details")

    user_data = {}
    for col in X.columns:
        if data[col].dtype in [np.int64, np.float64]:
            user_data[col] = st.sidebar.number_input(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
        else:
            options = data[col].unique().tolist()
            user_data[col] = st.sidebar.selectbox(f"{col}", options)

    user_df = pd.DataFrame([user_data])

    # Handle 'Yes'/'No' columns in user input
    for col in ['HasMortgage', 'HasCoSigner']:
         if col in user_df.columns:
             user_df[col] = user_df[col].map({'Yes': 1, 'No': 0})


    # Encode user input (for categorical features)
    for col in user_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        user_df[col] = le.fit_transform(user_df[col])


    # Scale input
    user_scaled = scaler.transform(user_df)

    # ------------------------------
    # Prediction
    # ------------------------------
    prediction = model.predict(user_scaled)
    prediction_proba = model.predict_proba(user_scaled)

    st.subheader("🔮 Prediction Result")
    result = "❌ Default" if prediction[0] == 1 else "✅ No Default"
    st.write(f"*Prediction:* {result}")

    st.subheader("📈 Prediction Probability")
    st.write(prediction_proba)
else:
    st.info("👆 Please upload a CSV file to start.")
