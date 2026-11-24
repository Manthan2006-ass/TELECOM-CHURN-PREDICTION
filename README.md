# Telecom Customer Churn & Retention Intelligence Engine

An enterprise-grade machine learning platform for customer churn risk prediction and retention intelligence. Built with a 10-Fold Blended Ensemble (LightGBM, CatBoost, and Semi-Supervised Pseudo-Labeling), calibrated for top-decile business capture and served via a FastAPI microservice and Streamlit retention dashboard.

---

## Measurable Results & Delivery

Evaluated on **601,237 customer records** via 10-fold stratified cross-validation with out-of-fold (`oof`) validation:

- **ROC-AUC**: **0.9158 (91.58%)** on the blended 3-way ensemble (0.40 LGB + 0.30 CatBoost + 0.30 Pseudo-LGBM).
- **Top Decile Capture Rate**: Identifies **36.73% of all churners** in the top 10% risk tier (49,836 out of 135,686 total churners).
- **Top Decile Precision & Lift**: **82.89% precision** (vs 22.57% base rate), delivering a **3.67x lift** over random retention outreach.
- **Top 20% Capture Rate**: Captures **63.48% of all churners** within the top 2 deciles.
- **Optimal Decision Threshold ($p = 0.36$)**: Achieves **0.7017 F1-score** and lifts recall to **78.09%**.
- **Model Calibration**: Brier score of **0.0955**.

---

## System Architecture

- **Machine Learning Pipeline**:
  - `utils.py`: Domain interaction feature engineering (`Charges_Ratio`, tenure hazard estimators), categorical alignment, and weighted ensemble logic.
  - `trainlgb.py`: 10-Fold Stratified LightGBM training.
  - `traincat.py`: 10-Fold Stratified CatBoost training.
  - `trainpsuedo.py`: Semi-supervised pseudo-labeling with confidence bounds ($>0.92$ and $<0.08$).
  - `ensemble.py`: Optimal OOF blending and master submission generation.
- **FastAPI Microservice (`api.py`)**:
  - Asynchronous lifespan architecture loading all 30 fold models.
  - Single-profile scoring endpoint (`/predict`) with business threshold parameters and rule-based explainability drivers.
  - High-throughput batch scoring endpoint (`/predict-batch`) with Monthly Recurring Revenue (MRR) at-risk analytics.
- **Streamlit Analytics Dashboard (`app.py`)**:
  - Cached model pipeline (`@st.cache_resource`) for zero-latency slider and dropdown interactions.
  - Single-customer decisioning summary with primary churn risk factor cards.
  - Batch dataset evaluation with interactive probability decile curves and downloadable retention queues.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Microservice
```bash
uvicorn api:app --reload --port 8000
```

### 3. Launch the Streamlit Retention Dashboard
```bash
streamlit run app.py
```
