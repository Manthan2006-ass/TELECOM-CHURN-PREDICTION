import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils import OUTPUT_DIR, load_data, CONFIG

def run_ensembling():
    weights = CONFIG["weights"]
    
    models = {
        "lgb": {
            "oof": f"{OUTPUT_DIR}/oof_lgb.csv",
            "sub": f"{OUTPUT_DIR}/submission_lgb.csv",
            "weight": weights["lgb"],
        },
        "cat": {
            "oof": f"{OUTPUT_DIR}/oof_cat.csv",
            "sub": f"{OUTPUT_DIR}/submission_cat.csv",
            "weight": weights["cat"],
        },
        "pseudo": {
            "oof": f"{OUTPUT_DIR}/oof_pseudo.csv",
            "sub": f"{OUTPUT_DIR}/submission_pseudo.csv",
            "weight": weights["pseudo"],
        },
    }

    for name, files in models.items():
        if not os.path.exists(files["oof"]) or not os.path.exists(files["sub"]):
            raise FileNotFoundError(f"Missing files for model '{name}'. Please run train modules first!")

    oof_data = {}
    min_rows = float("inf")

    for name, config in models.items():
        oof_data[name] = pd.read_csv(config["oof"])["oof"].values
        row_count = len(oof_data[name])
        if row_count < min_rows:
            min_rows = row_count

    _, y_true_series, _ = load_data()
    y_true = y_true_series.values[:min_rows]

    final_oof = np.zeros(min_rows)
    for name, config in models.items():
        aligned_oof_vals = oof_data[name][:min_rows]
        final_oof += aligned_oof_vals * config["weight"]

    ensemble_auc = roc_auc_score(y_true, final_oof)
    print(f"FINAL BLENDED ENSEMBLE OOF AUC: {ensemble_auc:.5f}")

    sample_sub = pd.read_csv(models["lgb"]["sub"])
    final_test_preds = np.zeros(len(sample_sub))

    for name, config in models.items():
        sub_df = pd.read_csv(config["sub"])
        final_test_preds += sub_df["Churn"].values * config["weight"]

    final_submission = pd.DataFrame({"id": sample_sub["id"], "Churn": final_test_preds})
    final_sub_path = f"{OUTPUT_DIR}/final_master_submission.csv"
    final_submission.to_csv(final_sub_path, index=False)
    print(f"Success! Master final prediction saved to: {final_sub_path}")

if __name__ == "__main__":
    run_ensembling()
