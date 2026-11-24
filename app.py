import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from utils import (
    add_interaction_features, 
    transform_categorical_inference, 
    load_master_models,
    predict_master_ensemble,
    explain_customer_risk,
    OPTIMAL_THRESHOLD,
    CONFIG
)

# Configure Streamlit page layout
st.set_page_config(
    page_title="Telco Churn Predictive & Retention Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model Registry Caching
@st.cache_resource(show_spinner=False)
def get_cached_models():
    """Caches loaded models in memory across user interactions."""
    return load_master_models()

master_models = get_cached_models()
lgb_models = master_models.get("lgb", [])
cat_models = master_models.get("cat", [])
pseudo_models = master_models.get("pseudo", [])
total_loaded_folds = len(lgb_models) + len(cat_models) + len(pseudo_models)

API_URL_SINGLE = "http://localhost:8000/predict"
API_URL_BATCH = "http://localhost:8000/predict-batch"

# Header section
st.title("Telecom Customer Churn & Retention Intelligence Engine")
st.caption("Enterprise 10-Fold Blended Ensemble (LightGBM + CatBoost + Pseudo-Labeling) | Evaluated on 601k+ Customer Records (ROC-AUC: 0.9158)")

# Sidebar: Controls & Input
st.sidebar.header("Decision Boundary & Calibration")
threshold_mode = st.sidebar.radio(
    "Decision Strategy",
    options=["Optimal F1-Score (36% - Recommended)", "Conservative (50%)", "Custom"],
    help="Targeting the top deciles with a calibrated 0.36 threshold captures 78.1% of all churners."
)

if threshold_mode == "Optimal F1-Score (36% - Recommended)":
    decision_threshold = OPTIMAL_THRESHOLD
elif threshold_mode == "Conservative (50%)":
    decision_threshold = 0.50
else:
    decision_threshold = st.sidebar.slider("Custom Churn Cutoff", min_value=0.10, max_value=0.90, value=OPTIMAL_THRESHOLD, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("Single Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner status", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents status", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure Length (Months)", min_value=0, max_value=72, value=12)

# Dynamic dropdown synchronization
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
total_charges = st.sidebar.number_input("Total Accumulation Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

# Single Customer Scoring
if st.sidebar.button("Run Churn Risk Analysis", use_container_width=True, type="primary"):
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
    
    final_prob = None
    risk_status = "UNKNOWN"
    risk_tier = "Unknown"
    rec = ""
    factors = []
    
    # Try FastAPI endpoint first
    try:
        res = requests.post(f"{API_URL_SINGLE}?threshold={decision_threshold}", json=payload, timeout=3)
        if res.status_code == 200:
            data = res.json()
            final_prob = data["churn_probability"]
            risk_status = data["risk_status"]
            risk_tier = data.get("risk_tier", "High Risk" if final_prob >= decision_threshold else "Safe")
            rec = data["recommendation"]
            factors = data.get("key_risk_factors", [])
        else:
            raise Exception("API error")
    except Exception:
        # Fallback to local master weighted ensemble
        if total_loaded_folds == 0:
            st.sidebar.error("Error: Models are not found on disk. Please train models first!")
        else:
            single_raw = pd.DataFrame([payload])
            proc_X = add_interaction_features(single_raw)
            encoded_df = transform_categorical_inference(proc_X)
            
            prob_arr = predict_master_ensemble(encoded_df, models_dict=master_models)
            final_prob = float(prob_arr[0])
            is_high = final_prob >= decision_threshold
            risk_status = "HIGH RISK" if is_high else "SAFE"
            
            if final_prob < 0.20:
                risk_tier = "Low Risk / Highly Loyal"
                rec = "Customer demonstrates high account stickiness. Standard relationship management."
            elif final_prob < decision_threshold:
                risk_tier = "Moderate Risk / Watchlist"
                rec = "Customer exhibits mild churn signals. Recommend engagement survey or low-cost perk."
            elif final_prob < 0.65:
                risk_tier = "High Risk / Action Required"
                rec = "Route profile to Retention Specialists. Present contract upgrade incentives."
            else:
                risk_tier = "Critical Risk / Urgent Intervention"
                rec = "Immediate executive escalation or personalized price-lock discount required."
                
            factors = explain_customer_risk(payload, final_prob)

    if final_prob is not None:
        with st.sidebar.container(border=True):
            st.subheader("Decisioning Summary")
            pct = final_prob * 100
            if final_prob >= decision_threshold:
                st.error(f"**{risk_status} ({pct:.1f}%)**")
            else:
                st.success(f"**{risk_status} ({pct:.1f}%)**")
                
            st.caption(f"Risk Tier: **{risk_tier}** (Cutoff: {decision_threshold:.2f})")
            st.write(f"**Action Plan:** {rec}")
            
            if factors:
                st.markdown("#### Primary Drivers")
                for f in factors:
                    icon = "🔴" if f["impact"] == "HIGH RISK" else ("🟡" if f["impact"] == "MODERATE RISK" else "🟢")
                    st.markdown(f"{icon} **{f['feature']}**: {f['detail']}")

# Main Page Layout: System Status & Batch Analysis
main_layout, status_panel = st.columns([2.2, 1])

with status_panel:
    with st.container(border=True):
        st.subheader("Pipeline Health & Registry")
        st.write(f"LightGBM Folds: **{len(lgb_models)} / 10**")
        st.write(f"CatBoost Folds: **{len(cat_models)} / 10**")
        st.write(f"Pseudo-LGBM Folds: **{len(pseudo_models)} / 10**")
        st.write(f"Ensemble Blend: **40% LGB + 30% CAT + 30% Pseudo**")
        st.write(f"Calibrated Threshold: **{decision_threshold:.2f}**")
        engine_status = "Fully Operational" if total_loaded_folds >= 10 else "Degraded / Train Models"
        st.caption(f"Status: **{engine_status}**")

with main_layout:
    st.header("Enterprise Batch Scoring & Financial Exposure")
    st.write("Upload client datasets to run feature engineering and score accounts with the 30-model ensemble.")
    
    with st.expander("CSV Format Guide & Template"):
        st.markdown("""
        Ensure your CSV contains columns like `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, etc.
        Missing columns will be handled with domain-safe imputation.
        """)
        template_df = pd.DataFrame([{
            "id": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
        }])
        st.download_button("Download CSV Template", data=template_df.to_csv(index=False).encode('utf-8'), file_name="churn_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload Raw CSV Dataset", type=["csv"])

# Batch Execution Handling
if uploaded_file is not None:
    st.markdown("---")
    st.header("Batch Processing & Risk Exposure Analysis")
    test_raw = pd.read_csv(uploaded_file)
    st.write(f"Loaded **{len(test_raw):,} records**. Sample preview:", test_raw.head(3))

    if total_loaded_folds == 0:
        st.error("Cannot score batch dataset: No trained models found in the outputs directory.")
    else:
        with st.spinner("Executing feature engineering & master ensemble scoring..."):
            try:
                # Try batch API first
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                res = requests.post(f"{API_URL_BATCH}?threshold={decision_threshold}", files=files, timeout=45)
                if res.status_code == 200:
                    batch_res = res.json()
                    live_predictions = batch_res["predictions"]
                    ids = batch_res["ids"]
                else:
                    raise Exception("Batch API offline")
            except Exception:
                # Fallback to local master ensemble
                mock_X = test_raw.copy().drop(columns=["id", "Churn"], errors="ignore")
                X_feat = add_interaction_features(mock_X)
                encoded_df = transform_categorical_inference(X_feat)
                live_predictions = predict_master_ensemble(encoded_df, models_dict=master_models).tolist()
                ids = test_raw["id"].tolist() if "id" in test_raw.columns else [f"CUST-{i}" for i in range(len(test_raw))]

        # Construct Output DataFrame
        output_df = pd.DataFrame({
            "id": ids,
            "Churn Probability": [round(p, 4) for p in live_predictions],
            "Risk Category": ["HIGH RISK" if p >= decision_threshold else "SAFE" for p in live_predictions],
            "Risk Tier": [
                "Critical Risk" if p >= 0.65 else ("High Risk" if p >= decision_threshold else ("Watchlist" if p >= 0.20 else "Low Risk"))
                for p in live_predictions
            ]
        })

        # Calculate Financial Exposure (MRR at Risk)
        high_risk_mask = output_df["Risk Category"] == "HIGH RISK"
        high_risk_count = int(high_risk_mask.sum())
        total_records = len(output_df)
        churn_rate = (high_risk_count / total_records) * 100 if total_records > 0 else 0

        total_mrr_risk = 0.0
        if "MonthlyCharges" in test_raw.columns:
            m_charges = pd.to_numeric(test_raw["MonthlyCharges"], errors="coerce").fillna(0.0)
            total_mrr_risk = float(m_charges[high_risk_mask].sum())

        # Metric KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Scored Accounts", f"{total_records:,}")
        kpi2.metric("At-Risk Accounts", f"{high_risk_count:,}", f"{churn_rate:.1f}%")
        kpi3.metric("Decision Cutoff", f"{decision_threshold:.2f}")
        kpi4.metric("MRR at Risk", f"${total_mrr_risk:,.2f}")

        # Visual Analytics
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk Tier Distribution")
            tier_counts = output_df["Risk Tier"].value_counts().reset_index()
            tier_counts.columns = ["Risk Tier", "Accounts"]
            st.bar_chart(tier_counts, x="Risk Tier", y="Accounts")

        with c2:
            st.subheader("Churn Probability Decile Curve")
            deciles = pd.qcut(output_df["Churn Probability"], q=10, duplicates="drop").value_counts().sort_index()
            decile_df = pd.DataFrame({"Probability Range": [str(interval) for interval in deciles.index], "Count": deciles.values})
            st.bar_chart(decile_df, x="Probability Range", y="Count")

        # Table & Download
        st.subheader("Actionable Retention Queue")
        filter_tier = st.selectbox("Filter by Risk Tier", ["All Accounts", "High & Critical Risk Only", "Critical Risk Only", "Safe Accounts Only"])
        if filter_tier == "High & Critical Risk Only":
            display_df = output_df[output_df["Risk Category"] == "HIGH RISK"]
        elif filter_tier == "Critical Risk Only":
            display_df = output_df[output_df["Risk Tier"] == "Critical Risk"]
        elif filter_tier == "Safe Accounts Only":
            display_df = output_df[output_df["Risk Category"] == "SAFE"]
        else:
            display_df = output_df

        st.dataframe(display_df, use_container_width=True, height=250)

        output_csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Scored Master Churn Predictions (CSV)",
            data=output_csv,
            file_name="master_churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
