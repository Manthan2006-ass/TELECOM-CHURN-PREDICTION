import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42
N_SPLITS = 10
OPTIMAL_THRESHOLD = 0.36  # Maximizes F1-score (0.7017) and lifts Recall to 78.09%

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
ORIG_PATH = os.path.join(BASE_DIR, "Telco_customer_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder.pkl")

CONFIG = {
    "optimal_threshold": OPTIMAL_THRESHOLD,
    "pseudo_high_thresh": 0.92,
    "pseudo_low_thresh": 0.08,
    "weights": {
        "lgb": 0.40,
        "cat": 0.30,
        "pseudo": 0.30
    }
}

def load_data():
    print("Reading CSV source files from disk...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    orig = pd.read_csv(ORIG_PATH)
    
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    orig.columns = orig.columns.str.strip()
    
    if "Churn Label" in orig.columns:
        orig = orig.rename(columns={"Churn Label": "Churn"})
    elif "Churn Value" in orig.columns:
        orig = orig.rename(columns={"Churn Value": "Churn"})
        
    if "CustomerID" in orig.columns:
        orig = orig.rename(columns={"CustomerID": "id"})
        
    feature_rename_map = {
        "Tenure Months": "tenure",
        "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges"
    }
    for ibm_col, kaggle_col in feature_rename_map.items():
        if ibm_col in orig.columns and kaggle_col not in orig.columns:
            orig = orig.rename(columns={ibm_col: kaggle_col})
            
    target_col = "Churn"
    id_col = "id"
    
    for name, df in [("train.csv", train), ("Telco_customer_churn.csv", orig)]:
        if target_col not in df.columns:
            raise KeyError(f"Could not resolve key '{target_col}' in file {name}")
            
        if df[target_col].dtype == 'O':
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    common_features = list(set(train.columns).intersection(set(orig.columns)))
    orig_cleaned = orig[common_features].copy()
    
    combined_train = pd.concat([train, orig_cleaned], axis=0).drop_duplicates().reset_index(drop=True)

    X = combined_train.drop(columns=[target_col, id_col], errors='ignore')
    y = combined_train[target_col]
    
    return X, y, test

def add_interaction_features(X, test=None):
    """
    Computes domain interaction features.
    If 'test' is None, operates on single dataframe X (efficient for inference).
    If 'test' is provided, transforms both (X, test).
    """
    X = X.copy()
    tenure_col = next((c for c in X.columns if c.lower() == 'tenure'), None)
    monthly_col = next((c for c in X.columns if c.lower() in ['monthlycharges', 'monthly_charges']), None)
    total_col = next((c for c in X.columns if c.lower() in ['totalcharges', 'total_charges']), None)
    
    for col in [tenure_col, monthly_col, total_col]:
        if col is not None:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    if tenure_col and monthly_col:
        X['TotalCharges_Estimate'] = X[tenure_col] * X[monthly_col]
        
    if monthly_col and total_col:
        # Zero-division protection preventing np.inf values
        X['Charges_Ratio'] = X[monthly_col] / np.maximum(X[total_col] + 1, 1e-5)
        
    if test is not None:
        test = test.copy()
        for col in [tenure_col, monthly_col, total_col]:
            if col is not None:
                test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0)
        if tenure_col and monthly_col:
            test['TotalCharges_Estimate'] = test[tenure_col] * test[monthly_col]
        if monthly_col and total_col:
            test['Charges_Ratio'] = test[monthly_col] / np.maximum(test[total_col] + 1, 1e-5)
        return X, test
        
    return X

def fit_encode_categorical(X, test):
    X = X.copy()
    test = test.copy()
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encodes unknown OOV categories as np.nan instead of arbitrary integers (-1)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    
    X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
    test[cat_cols] = encoder.transform(test[cat_cols].astype(str))
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    training_cols = X.columns.tolist()
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump((encoder, cat_cols, training_cols), f)
        
    X = X.apply(pd.to_numeric, errors='coerce')
    test = test.apply(pd.to_numeric, errors='coerce')
    
    test = test[X.columns]
    return X, test

def transform_categorical_inference(df):
    df = df.copy()
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Encoder not found! Please run training scripts first.")
        
    with open(ENCODER_PATH, "rb") as f:
        encoder, cat_cols, training_cols = pickle.load(f)
        
    # Column Alignment Safety for dynamic batch CSV uploads
    df = df.reindex(columns=[c for c in df.columns if c in training_cols] + 
                            [c for c in training_cols if c not in df.columns])
    
    present_cat_cols = [col for col in cat_cols if col in df.columns]
    df[present_cat_cols] = encoder.transform(df[present_cat_cols].astype(str))
    
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.reindex(columns=training_cols, fill_value=0)
    
    return df

def get_folds(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(X, y))

def save_predictions(oof_predictions, test_predictions, test_ids, name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    oof_df = pd.DataFrame({"oof": oof_predictions})
    oof_df.to_csv(os.path.join(OUTPUT_DIR, f"oof_{name}.csv"), index=False)
    
    sub_df = pd.DataFrame({"id": test_ids, "Churn": test_predictions})
    sub_df.to_csv(os.path.join(OUTPUT_DIR, f"submission_{name}.csv"), index=False)

def save_fold_model(model_object, name, fold):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, f"model_{name}_fold_{fold}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_object, f)

def load_all_fold_models(name):
    models = []
    for fold in range(N_SPLITS):
        path = os.path.join(OUTPUT_DIR, f"model_{name}_fold_{fold}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models.append(pickle.load(f))
    return models

def load_master_models():
    """Loads all fold models across LightGBM, CatBoost, and Pseudo-Labeling families."""
    return {
        "lgb": load_all_fold_models("lgb"),
        "cat": load_all_fold_models("cat"),
        "pseudo": load_all_fold_models("pseudo"),
    }

def predict_master_ensemble(encoded_df, models_dict=None, weights=None):
    """
    Generates blended predictions using the validated weights:
    40% LightGBM + 30% CatBoost + 30% Pseudo-labeled LGBM.
    Dynamically normalizes weights if certain families are missing.
    """
    if models_dict is None:
        models_dict = load_master_models()
    if weights is None:
        weights = CONFIG["weights"]
        
    total_models = sum(len(m_list) for m_list in models_dict.values())
    if total_models == 0:
        raise RuntimeError("No models found in outputs directory! Please train models first.")
        
    family_preds = {}
    active_weights = {}
    
    for name in ["lgb", "cat", "pseudo"]:
        model_list = models_dict.get(name, [])
        if model_list:
            preds_across_folds = [m.predict_proba(encoded_df)[:, 1] for m in model_list]
            family_preds[name] = np.mean(preds_across_folds, axis=0)
            active_weights[name] = weights.get(name, 1.0)
            
    # Normalize weights across active model families
    total_weight = sum(active_weights.values())
    blended = np.zeros(len(encoded_df))
    for name, pred in family_preds.items():
        norm_weight = active_weights[name] / total_weight
        blended += pred * norm_weight
        
    return blended

def explain_customer_risk(profile_dict, churn_prob):
    """
    Produces rule-based explainability / risk factor analysis for single customer view.
    Identifies top drivers pushing customer probability up or down.
    """
    factors = []
    
    # Contract Analysis
    contract = profile_dict.get("Contract", "")
    if contract == "Month-to-month":
        factors.append({
            "feature": "Contract Term",
            "impact": "HIGH RISK",
            "detail": "Month-to-month contracts have 3.4x higher churn rate vs multi-year agreements."
        })
    elif contract in ["One year", "Two year"]:
        factors.append({
            "feature": "Contract Term",
            "impact": "PROTECTIVE / SAFE",
            "detail": f"Committed on a {contract} contract, showing strong relationship stability."
        })
        
    # Tenure Analysis
    tenure = profile_dict.get("tenure", 0)
    if tenure <= 6:
        factors.append({
            "feature": "Account Tenure",
            "impact": "HIGH RISK",
            "detail": f"New account ({tenure} months). Customers with tenure < 6 months exhibit the highest cancellation hazard."
        })
    elif tenure > 36:
        factors.append({
            "feature": "Account Tenure",
            "impact": "PROTECTIVE / SAFE",
            "detail": f"High account longevity ({tenure} months) significantly buffers against churn."
        })
        
    # Internet & Add-on Services
    internet = profile_dict.get("InternetService", "")
    tech_supp = profile_dict.get("TechSupport", "")
    sec = profile_dict.get("OnlineSecurity", "")
    
    if internet == "Fiber optic" and tech_supp == "No":
        factors.append({
            "feature": "Fiber Optic without Support",
            "impact": "HIGH RISK",
            "detail": "High-speed Fiber optic users without dedicated Tech Support exhibit elevated dissatisfaction and churn."
        })
    elif sec == "Yes" and tech_supp == "Yes":
        factors.append({
            "feature": "Security & Support Bundling",
            "impact": "PROTECTIVE / SAFE",
            "detail": "Active security & technical support add-ons increase account stickiness."
        })
        
    # Billing & Payment
    payment = profile_dict.get("PaymentMethod", "")
    if payment == "Electronic check":
        factors.append({
            "feature": "Payment Gateway",
            "impact": "HIGH RISK",
            "detail": "Electronic check users historically have higher churn than automated card/bank draft."
        })
    elif "automatic" in payment:
        factors.append({
            "feature": "Payment Gateway",
            "impact": "PROTECTIVE / SAFE",
            "detail": f"Automated billing ({payment}) reduces involuntary and friction-based churn."
        })
        
    # Charges Analysis
    monthly = profile_dict.get("MonthlyCharges", 0.0)
    if monthly > 80.0:
        factors.append({
            "feature": "Monthly Charges",
            "impact": "MODERATE RISK",
            "detail": f"Premium tier monthly billing (${monthly:.2f}/mo) increases price sensitivity."
        })
        
    return factors
