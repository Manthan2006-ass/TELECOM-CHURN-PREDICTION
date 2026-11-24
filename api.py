import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from schemas import CustomerProfileSchema, PredictionOutputSchema
from utils import (
    add_interaction_features, 
    transform_categorical_inference, 
    load_master_models,
    predict_master_ensemble,
    explain_customer_risk,
    OPTIMAL_THRESHOLD,
    CONFIG
)

# Application state holding loaded model registry
model_registry = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_registry
    print("Loading Master Ensemble Model Registry (LightGBM + CatBoost + Pseudo-LGBM)...")
    model_registry = load_master_models()
    lgb_count = len(model_registry.get("lgb", []))
    cat_count = len(model_registry.get("cat", []))
    pseudo_count = len(model_registry.get("pseudo", []))
    print(f"Registry online: {lgb_count} LightGBM, {cat_count} CatBoost, {pseudo_count} Pseudo-LGBM folds loaded.")
    yield
    model_registry.clear()

app = FastAPI(
    title="Telco Churn Enterprise ML Microservice",
    description="Production-grade API serving 10-Fold Blended Ensembles with Explainability & Business Calibration",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "lgb_folds_loaded": len(model_registry.get("lgb", [])),
        "cat_folds_loaded": len(model_registry.get("cat", [])),
        "pseudo_folds_loaded": len(model_registry.get("pseudo", [])),
        "weights": CONFIG["weights"]
    }

@app.post("/predict", response_model=PredictionOutputSchema)
def predict_churn(
    payload: CustomerProfileSchema,
    threshold: float = Query(OPTIMAL_THRESHOLD, ge=0.05, le=0.95, description="Decision probability threshold")
):
    if not any(len(m_list) > 0 for m_list in model_registry.values()):
        raise HTTPException(status_code=503, detail="Model registry is offline or not trained.")

    try:
        raw_dict = payload.model_dump()
        raw_df = pd.DataFrame([raw_dict])
        
        # Domain interaction features on single record
        proc_X = add_interaction_features(raw_df)
        encoded_df = transform_categorical_inference(proc_X)

        # Execute 3-way weighted master ensemble
        prob_arr = predict_master_ensemble(encoded_df, models_dict=model_registry)
        final_prob = float(prob_arr[0])
        
        is_high_risk = final_prob >= threshold

        # Segment into granular risk tiers
        if final_prob < 0.20:
            risk_tier = "Low Risk / Highly Loyal"
            recommendation = "Customer demonstrates high account stickiness. Standard relationship management."
        elif final_prob < threshold:
            risk_tier = "Moderate Risk / Watchlist"
            recommendation = "Customer exhibits mild churn signals. Recommend engagement survey or low-cost perk."
        elif final_prob < 0.65:
            risk_tier = "High Risk / Action Required"
            recommendation = "Route profile to Retention Specialists. Present contract upgrade incentives."
        else:
            risk_tier = "Critical Risk / Urgent Intervention"
            recommendation = "Immediate executive escalation or personalized price-lock discount required."

        # Risk drivers for explainability
        factors = explain_customer_risk(raw_dict, final_prob)

        return PredictionOutputSchema(
            churn_probability=round(final_prob, 4),
            risk_status="HIGH RISK" if is_high_risk else "SAFE",
            risk_tier=risk_tier,
            decision_threshold=round(threshold, 2),
            recommendation=recommendation,
            key_risk_factors=factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    threshold: float = Query(OPTIMAL_THRESHOLD, ge=0.05, le=0.95)
):
    if not any(len(m_list) > 0 for m_list in model_registry.values()):
        raise HTTPException(status_code=503, detail="Model registry is offline or not trained.")
    
    try:
        test_raw = pd.read_csv(file.file)
        if test_raw.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        mock_X = test_raw.copy().drop(columns=["id", "Churn"], errors="ignore")
        
        # Single dataframe feature computation
        X_feat = add_interaction_features(mock_X)
        encoded_df = transform_categorical_inference(X_feat)

        # Batch scoring via master ensemble
        batch_probs = predict_master_ensemble(encoded_df, models_dict=model_registry).tolist()
        
        # Calculate business analytics
        high_risk_flags = [p >= threshold for p in batch_probs]
        high_risk_count = sum(high_risk_flags)
        
        # Revenue at risk estimation if MonthlyCharges is present
        total_mrr_at_risk = 0.0
        if "MonthlyCharges" in test_raw.columns:
            m_charges = pd.to_numeric(test_raw["MonthlyCharges"], errors="coerce").fillna(0.0)
            total_mrr_at_risk = float(m_charges[pd.Series(high_risk_flags)].sum())

        return {
            "status": "success",
            "total_records": len(batch_probs),
            "high_risk_count": high_risk_count,
            "churn_rate_predicted": round(high_risk_count / len(batch_probs), 4),
            "total_mrr_at_risk": round(total_mrr_at_risk, 2),
            "decision_threshold": threshold,
            "predictions": [round(p, 4) for p in batch_probs],
            "risk_categories": ["HIGH RISK" if f else "SAFE" for f in high_risk_flags],
            "ids": test_raw["id"].tolist() if "id" in test_raw.columns else [f"CUST-{i}" for i in range(len(test_raw))]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")
