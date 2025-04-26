import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import io

# Title of the app
st.title("🚨 AI-Based Intrusion Detection System")

# Explanation about the file upload
st.markdown("""
    Upload the **NSL-KDD** dataset in CSV or TXT format. This dataset will be used to train a machine learning model 
    to detect network intrusions. The model will help identify whether network traffic is **normal** or if it 
    contains an **attack**.
""")

# ✅ File upload with description
uploaded_file = st.file_uploader("Upload NSL-KDD Dataset (.csv or .txt)", type=['csv', 'txt'], key="file_uploader_1")

# Define columns for NSL-KDD dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = columns
    df = df.drop("difficulty", axis=1)  # Drop 'difficulty' column

    st.write("### 📄 Raw Data Preview", df.head())

    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'label':
            df[col] = label_encoder.fit_transform(df[col])

    # Encode label column separately
    df["label"] = label_encoder.fit_transform(df["label"])

    # Features and target variable
    X = df.drop("label", axis=1)
    y = df["label"]

    # Split the data into training and testing sets, and scale the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.write(f"✅ Shape of X_train: {X_train.shape}")
    st.write(f"✅ Shape of X_test: {X_test.shape}")

    # Load saved model and scaler if available, else train a new model
    if os.path.exists('random_forest_model.pkl') and os.path.exists('scaler.pkl'):
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        st.write("✅ Loaded pre-trained model and scaler.")
    else:
        # Train a RandomForest model
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)
        joblib.dump(model, 'random_forest_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        st.write("✅ Model trained and saved successfully.")

    # Predict on the test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.write("### ✅ Model Accuracy")
    st.progress(acc)  # Visual progress bar for model accuracy
    st.write(f"Model Accuracy: {acc:.2f}")

    # Display confusion matrix and classification report
    st.write("### 📊 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("### 📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # SHAP Summary Plot
    st.write("### 🔍 SHAP Explanation (Top 100 Samples)")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test[:100], check_additivity=False)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test[:100], feature_names=X.columns, show=False)
    st.pyplot(fig)

    # Real-time Intrusion Detection Section
    st.write("### 🕵️ Real-time Intrusion Detection")
    with st.expander("🔧 Enter input for new detection"):
        input_data = {}
        for col in X.columns:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                input_data[col] = st.number_input(f"Enter value for {col}", value=0.0, key=f"{col}_input")
            else:
                input_data[col] = st.selectbox(f"Enter value for {col}", df[col].unique(), key=f"{col}_input")

        if st.button("🚀 Predict Intrusion"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            # Feedback for users based on prediction
            if predicted_label == 'normal':
                st.success("✅ The input is classified as **NORMAL**. Your network is secure.")
            else:
                st.error("🚨 The input is classified as an **ATTACK**! Please check your network for unusual activity.")

            # Save input data to CSV
            save_data = st.button("💾 Save Input Data")
            if save_data:
                # Convert the input data to a DataFrame and save to CSV
                input_data_df = pd.DataFrame([input_data])
                input_data_df.to_csv('input_data.csv', mode='a', header=not os.path.exists('input_data.csv'), index=False)
                st.success("✅ Input data saved successfully as 'input_data.csv'.")

            # Prepare the report for download
            report = f"Model Accuracy: {acc:.2f}\n\n"
            report += f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n\n"
            report += f"Classification Report:\n{classification_report(y_test, y_pred)}\n\n"
            report += f"SHAP Summary Plot: (Saved as image)\n"

            # Save SHAP plot as image
            shap.summary_plot(shap_values, X_test[:100], feature_names=X.columns, show=False)
            shap_image_path = "shap_summary_plot.png"
            plt.savefig(shap_image_path)

            # Create a downloadable report (text and image)
            report_file = io.BytesIO()
            report_file.write(report.encode())
            report_file.seek(0)

            # Download button for the report (text file + image)
            st.download_button(
                label="Download Full Report",
                data=report_file,
                file_name="intrusion_detection_report.txt",
                mime="text/plain"
            )
            st.download_button(
                label="Download SHAP Summary Plot",
                data=open(shap_image_path, "rb").read(),
                file_name="shap_summary_plot.png",
                mime="image/png"
            )
else:
    st.warning("Please upload a dataset file to begin.")





















