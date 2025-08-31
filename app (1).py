import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("💳 Loan Default Prediction App")
st.write("Upload your dataset and predict if a loan applicant will default or not.")

# ------------------------------
# File Upload
# ------------------------------
st.subheader("⬆️ Upload Your Dataset")
uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(data.head())

    st.sidebar.header("🔧 Model Settings")
    target_col = st.sidebar.selectbox("Select Target Column (Default)", data.columns, index=len(data.columns)-1)

    # ------------------------------
    # Preprocessing
    # ------------------------------
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Encode target column if it's categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Handle missing values
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Detect and map binary 'Yes'/'No' columns
    binary_cols = [col for col in X.columns if sorted(X[col].dropna().unique().tolist()) == ['No', 'Yes']]
    for col in binary_cols:
        X[col] = X[col].map({'Yes': 1, 'No': 0})

    # Encode categorical columns
    encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le  # Save encoder for later

    # Feature Scaling
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

    st.write(f"✅ Model trained successfully! **Accuracy:** *{accuracy*100:.2f}%*")

    # ------------------------------
    # User Input
    # ------------------------------
    st.sidebar.header("📝 Enter Applicant Details")
    user_data = {}

    for col in X.columns:
        original_col = data[col] if col in data.columns else None

        if col in binary_cols:
            user_data[col] = st.sidebar.selectbox(f"{col}", ['Yes', 'No'])
        elif original_col is not None and original_col.dtype in [np.int64, np.float64]:
            user_data[col] = st.sidebar.number_input(
                f"{col}", 
                float(data[col].min()), 
                float(data[col].max()), 
                float(data[col].mean())
            )
        else:
            options = data[col].unique().tolist() if col in data.columns else []
            user_data[col] = st.sidebar.selectbox(f"{col}", options)

    user_df = pd.DataFrame([user_data])

    # Handle binary columns
    for col in binary_cols:
        if col in user_df.columns:
            user_df[col] = user_df[col].map({'Yes': 1, 'No': 0})

    # Encode categorical features with stored encoders
    for col, le in encoders.items():
        if col in user_df.columns:
            user_df[col] = le.transform([user_df[col][0]])

    # Feature scaling
    user_scaled = scaler.transform(user_df)

    # ------------------------------
    # Prediction
    # ------------------------------
    prediction = model.predict(user_scaled)
    prediction_proba = model.predict_proba(user_scaled)

    st.subheader("🔮 Prediction Result")
    if prediction[0] == 1:
        st.error("❌ Prediction: Applicant is likely to **Default** on the loan.")
    else:
        st.success("✅ Prediction: Applicant is **Not Likely to Default** on the loan.")

    st.subheader("📈 Prediction Probability")
    st.write(pd.DataFrame(prediction_proba, columns=["No Default", "Default"]))
else:
    st.info("👆 Please upload a CSV file to begin.")
