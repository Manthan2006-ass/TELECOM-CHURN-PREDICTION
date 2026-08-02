import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42
N_SPLITS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
ORIG_PATH = os.path.join(BASE_DIR, "Telco_customer_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder.pkl")

CONFIG = {
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

def add_interaction_features(X, test):
    X = X.copy()
    test = test.copy()
    
    tenure_col = next((c for c in X.columns if c.lower() == 'tenure'), None)
    monthly_col = next((c for c in X.columns if c.lower() in ['monthlycharges', 'monthly_charges']), None)
    total_col = next((c for c in X.columns if c.lower() in ['totalcharges', 'total_charges']), None)
    
    for col in [tenure_col, monthly_col, total_col]:
        if col is not None:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0)
    
    if tenure_col and monthly_col:
        X['TotalCharges_Estimate'] = X[tenure_col] * X[monthly_col]
        test['TotalCharges_Estimate'] = test[tenure_col] * test[monthly_col]
        
    if monthly_col and total_col:
        # FIX: Zero-division protection preventing np.inf values
        X['Charges_Ratio'] = X[monthly_col] / np.maximum(X[total_col] + 1, 1e-5)
        test['Charges_Ratio'] = test[monthly_col] / np.maximum(test[total_col] + 1, 1e-5)
        
    return X, test

def fit_encode_categorical(X, test):
    X = X.copy()
    test = test.copy()
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # FIX: Encodes unknown OOV categories as np.nan instead of arbitrary integers (-1)
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
        
    # FIX: Column Alignment Safety for dynamic batch CSV uploads
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
