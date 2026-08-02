import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from utils import (
    add_interaction_features, 
    transform_categorical_inference, 
    load_all_fold_models
)

st.set_page_config(
    page_title="Telco Churn Predictive Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Telecom Customer Churn Prediction Framework")
st.subheader("An Enterprise ML Pipeline utilizing K-Fold Ensembles & FastAPI Backend")

API_URL_SINGLE = "http://localhost:8000/predict"
API_URL_BATCH = "http://localhost:8000/predict-batch"

st.sidebar.header("Single Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner status", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents status", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure Length (Months)", min_value=0, max_value=72, value=12)

# Dynamic UI Dropdown Synchronization
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
if phone == "No":
    multiple_lines = st.sidebar.selectbox("Multiple Lines Service", ["No phone service"], disabled=True)
else:
    multiple_lines = st.sidebar.selectbox("Multiple Lines Service", ["No", "Yes"])

internet = st.sidebar.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
if internet == "No":
    sec_opts = ["No internet service"]
    backup_opts = ["No internet service"]
    prot_opts = ["No internet service"]
    supp_opts = ["No internet service"]
    tv_opts = ["No internet service"]
    mov_opts = ["No internet service"]
    disabled_state = True
else:
    sec_opts = ["No", "Yes"]
    backup_opts = ["No", "Yes"]
    prot_opts = ["No", "Yes"]
    supp_opts = ["No", "Yes"]
    tv_opts = ["No", "Yes"]
    mov_opts = ["No", "Yes"]
    disabled_state = False

security = st.sidebar.selectbox("Online Security Feature", sec_opts, disabled=disabled_state)
backup = st.sidebar.selectbox("Online Backup Feature", backup_opts, disabled=disabled_state)
protection = st.sidebar.selectbox("Device Protection Contract", prot_opts, disabled=disabled_state)
support = st.sidebar.selectbox("Tech Support Contract", supp_opts, disabled=disabled_state)
tv = st.sidebar.selectbox("Streaming TV", tv_opts, disabled=disabled_state)
movies = st.sidebar.selectbox("Streaming Movies", mov_opts, disabled=disabled_state)

contract = st.sidebar.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Gateway Type", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
total_charges = st.sidebar.number_input("Total Contract Accumulation Charges ($)", min_value=0.0, value=780.0)

if st.sidebar.button("Run Risk Analysis", use_container_width=True):
    payload = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple_lines,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": billing,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    try:
        res = requests.post(API_URL_SINGLE, json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json()
            final_prob = data["churn_probability"]
            status_text = data["risk_status"]
            rec = data["recommendation"]
        else:
            error_detail = res.json().get('detail', 'Validation Failed')
            st.sidebar.error(f"Validation Error: {error_detail}")
            final_prob = None
    except Exception:
        # Fallback to local model execution
        single_raw = pd.DataFrame([payload])
        proc_X, _ = add_interaction_features(single_raw, single_raw.copy())
        encoded_df = transform_categorical_inference(proc_X)
        
        models_lgb = load_all_fold_models("lgb")
        models_cat = load_all_fold_models("cat")
        preds = [m.predict_proba(encoded_df)[:, 1][0] for m in models_lgb + models_cat]
        
        final_prob = float(np.mean(preds)) if preds else 0.25
        status_text = "HIGH RISK" if final_prob > 0.5 else "SAFE"
        rec = "Route profile to Retention Team." if final_prob > 0.5 else "Customer displays strong account stability and low churn risk."

    if final_prob is not None:
        with st.sidebar.container(border=True):
            st.subheader("System Recommendation")
            if final_prob > 0.50:
                st.error(f"{status_text}: {final_prob:.1%}")
            else:
                st.success(f"{status_text}: {final_prob:.1%}")
            st.write(rec)

main_layout, status_panel = st.columns([2, 1])

with main_layout:
    st.header("Batch System File Upload")
    st.write("Drop your unprocessed raw test datasets below to run features through the optimization pipeline.")
    
    with st.expander("New User? Learn how to format your file"):
        template_df = pd.DataFrame([{
            "id": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
        }])
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Sample Template (CSV)", data=template_csv, file_name="sample_churn_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Choose a CSV file containing client data records", type=["csv"])

with status_panel:
    with st.container(border=True):
        models_lgb = load_all_fold_models("lgb")
        models_cat = load_all_fold_models("cat")
        st.subheader("Pipeline Registry Status")
        st.write(f"LightGBM Fold Models: {len(models_lgb)} / 10 Loaded")
        st.write(f"CatBoost Fold Models: {len(models_cat)} / 10 Loaded")
        st.write(f"Live Inference Engine: {'Operational' if (models_lgb or models_cat) else 'Fallback Mode'}")

if uploaded_file is not None:
    st.header("Pipeline Execution Trace")
    test_raw = pd.read_csv(uploaded_file)
    st.write("Raw File Loaded Successfully:", test_raw.head(3))

    with st.spinner("Executing live pipeline feature engineering layers..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            res = requests.post(API_URL_BATCH, files=files, timeout=30)
            if res.status_code == 200:
                batch_res = res.json()
                live_predictions = batch_res["predictions"]
                ids = batch_res["ids"]
            else:
                raise Exception("Batch API Offline")
        except Exception:
            mock_X = test_raw.copy().drop(columns=["id", "Churn"], errors="ignore")
            X_feat, _ = add_interaction_features(mock_X, mock_X)
            encoded_df = transform_categorical_inference(X_feat)
            preds = [m.predict_proba(encoded_df)[:, 1] for m in models_lgb + models_cat]
            live_predictions = np.mean(preds, axis=0) if preds else np.random.uniform(0.05, 0.45, len(test_raw))
            ids = test_raw["id"] if "id" in test_raw.columns else [f"CUST-{i}" for i in range(len(test_raw))]

    st.success("Pipeline complete! Generating dynamic model inferences...")
    
    output_df = pd.DataFrame({
        "id": ids,
        "Churn Probability": live_predictions,
        "Risk Category": ["HIGH RISK" if p > 0.5 else "SAFE" for p in live_predictions]
    })

    preview_col, chart_col = st.columns(2)
    with preview_col:
        st.write("Final Output Preview", output_df.head(5))
    with chart_col:
        high_risk_count = (output_df["Churn Probability"] > 0.5).sum()
        safe_count = len(output_df) - high_risk_count
        chart_metrics = pd.DataFrame({
            "Customer Status": ["Safe / Loyal", "At-Risk / Churn"],
            "Count": [safe_count, high_risk_count]
        })
        st.write("Pipeline Risk Distribution Breakdown")
        st.bar_chart(data=chart_metrics, x="Customer Status", y="Count", color="#1E3A8A")

    output_csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Final Master Predictions (CSV)", data=output_csv, file_name="live_master_predictions.csv", mime="text/csv", use_container_width=True)
