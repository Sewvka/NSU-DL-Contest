import polars as pl
import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

# Config
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed_v5")
RANDOM_SEED = 42
VAL_START = pd.Timestamp("2025-05-01")

CAT_COLS_MODEL = [
    "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd",
    "timezone", "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
    "developer_tools_i", "compromised_i",
]

META_COLS = ["event_id", "period", "event_ts", "is_train_sample", "is_inference", "train_target_raw", "target_bin", "event_date"]

def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def train_model():
    print("Loading data for GPU Training (V5 Checked)...")
    parts = [pl.read_parquet(PROCESSED_DIR / f"features_part_{i}.parquet") for i in [1, 2, 3]]
    df = pl.concat(parts).sort("event_ts")
    del parts; gc.collect()
    
    print("Converting to Pandas & Optimizing types...")
    train_all_df = df.filter(pl.col("is_train_sample")).to_pandas()
    test_df = df.filter(pl.col("is_inference")).to_pandas()
    del df; gc.collect()
    
    feature_cols = [c for c in train_all_df.columns if c not in META_COLS and c != "target"]
    num_cols = [c for c in feature_cols if c not in CAT_COLS_MODEL]
    
    # Оптимизация памяти: float64 -> float32
    for c in num_cols:
        train_all_df[c] = train_all_df[c].astype(np.float32)
        test_df[c] = test_df[c].astype(np.float32)

    for c in CAT_COLS_MODEL:
        train_all_df[c] = train_all_df[c].fillna(-1).astype(np.int64)
        test_df[c] = test_df[c].fillna(-1).astype(np.int64)

    train_all_df["event_ts"] = pd.to_datetime(train_all_df["event_ts"])
    test_df["event_ts"] = pd.to_datetime(test_df["event_ts"])

    val_mask = train_all_df["event_ts"] >= VAL_START
    
    # ЧЕСТНОЕ заполнение пропусков (считаем медиану только по X_tr)
    print("Imputing nulls (No Leakage)...")
    train_median = train_all_df.loc[~val_mask, num_cols].median()
    train_all_df[num_cols] = train_all_df[num_cols].fillna(train_median)
    test_df[num_cols] = test_df[num_cols].fillna(train_median)

    # Веса
    n_pos = (train_all_df["target_bin"] == 1).sum()
    n_neg = (train_all_df["target_bin"] == 0).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Параметры для GPU (6GB VRAM Safe)
    params = {
        "iterations": 2000, 
        "learning_rate": 0.05, 
        "depth": 8, 
        "eval_metric": "PRAUC",
        "random_seed": 42, 
        "verbose": 100, 
        "early_stopping_rounds": 200,
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.9, # Оставляем 10% видеопамяти системе
        "scale_pos_weight": min(scale_weight, 15)
    }
    
    print(f"\n--- Stage 1: Validation (Rows: {len(train_all_df)}) ---")
    X_tr, y_tr = train_all_df.loc[~val_mask, feature_cols], train_all_df.loc[~val_mask, "target_bin"]
    X_val, y_val = train_all_df.loc[val_mask, feature_cols], train_all_df.loc[val_mask, "target_bin"]
    
    model = CatBoostClassifier(**params)
    model.fit(Pool(X_tr, y_tr, cat_features=CAT_COLS_MODEL),
              eval_set=Pool(X_val, y_val, cat_features=CAT_COLS_MODEL))
    
    val_auc = average_precision_score(y_val, model.predict(X_val, prediction_type="RawFormulaVal"))
    print(f"\nValidation PR-AUC: {val_auc:.6f}")
    best_iter = model.get_best_iteration()
    
    # Очистка памяти перед Refit
    del X_tr, y_tr, X_val, y_val, model; gc.collect()

    print("\n--- Stage 2: Full Refit ---")
    final_model = CatBoostClassifier(**{**params, "iterations": best_iter or 1500, "verbose": 100})
    final_model.fit(Pool(train_all_df[feature_cols], train_all_df["target_bin"], cat_features=CAT_COLS_MODEL))
    
    print("Predicting...")
    preds = _sigmoid(final_model.predict(test_df[feature_cols], prediction_type="RawFormulaVal"))
    
    sub = pd.DataFrame({"event_id": test_df["event_id"].values.astype(np.int64), "predict": preds})
    sample_sub = pd.read_csv(DATA_DIR / "sample_submit.csv")
    final_sub = sample_sub[["event_id"]].merge(sub, on="event_id", how="left").fillna(0)
    final_sub.to_csv("submission_v5_gpu.csv", index=False)
    
    print(f"Success! Final submission saved. Non-zero predicts: {(final_sub['predict'] > 0.01).sum()}")

if __name__ == "__main__":
    train_model()
