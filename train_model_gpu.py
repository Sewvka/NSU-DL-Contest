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

META_COLS = ["event_id", "period", "event_ts", "is_train_sample", "is_inference", "train_target_raw", "target_bin", "event_date", "target"]

def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def train_model():
    print("Loading data for GPU Training (Low Memory Support)...")
    parts = [pl.read_parquet(PROCESSED_DIR / f"features_part_{i}.parquet") for i in [1, 2, 3]]
    df = pl.concat(parts)
    del parts; gc.collect()
    
    print("Converting to Pandas...")
    train_all_df = df.filter(pl.col("is_train_sample")).to_pandas()
    test_df = df.filter(pl.col("is_inference")).to_pandas()
    del df; gc.collect()
    
    # Автоматически определяем колонки
    feature_cols = [c for c in train_all_df.columns if c not in META_COLS]
    
    # Категориальные - те, что имеют тип object или int, и НЕ являются вещественными
    cat_cols = [c for c in feature_cols if "amt" not in c and "sec" not in c and "rate" not in c and "cnt" not in c]
    print(f"Features: {len(feature_cols)}, Categorical: {cat_cols}")

    for df_tmp in [train_all_df, test_df]:
        df_tmp["event_ts"] = pd.to_datetime(df_tmp["event_ts"])
        for c in cat_cols:
            df_tmp[c] = df_tmp[c].fillna(-1).astype(np.int64)
        for c in [f for f in feature_cols if f not in cat_cols]:
            df_tmp[c] = df_tmp[c].astype(np.float32).fillna(0)

    val_mask = train_all_df["event_ts"] >= VAL_START
    
    # Веса
    n_pos = (train_all_df["target"] == 1).sum()
    n_neg = (train_all_df["target"] == 0).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = {
        "iterations": 1500, 
        "learning_rate": 0.05, 
        "depth": 7, 
        "eval_metric": "PRAUC",
        "random_seed": 42, 
        "verbose": 100, 
        "early_stopping_rounds": 100,
        "task_type": "GPU",
        "devices": "0",
        "scale_pos_weight": min(scale_weight, 15)
    }
    
    print("\n--- Stage 1: Validation ---")
    X_tr, y_tr = train_all_df.loc[~val_mask, feature_cols], train_all_df.loc[~val_mask, "target"].fillna(0)
    X_val, y_val = train_all_df.loc[val_mask, feature_cols], train_all_df.loc[val_mask, "target"].fillna(0)
    
    model = CatBoostClassifier(**params)
    model.fit(Pool(X_tr, y_tr, cat_features=cat_cols),
              eval_set=Pool(X_val, y_val, cat_features=cat_cols))
    
    best_iter = model.get_best_iteration()
    del X_tr, y_tr, X_val, y_val, model; gc.collect()

    print("\n--- Stage 2: Refit ---")
    final_model = CatBoostClassifier(**{**params, "iterations": best_iter or 1000, "verbose": 100})
    final_model.fit(Pool(train_all_df[feature_cols], train_all_df["target"].fillna(0), cat_features=cat_cols))
    
    print("Predicting...")
    preds = _sigmoid(final_model.predict(test_df[feature_cols], prediction_type="RawFormulaVal"))
    
    sub = pd.DataFrame({"event_id": test_df["event_id"].values.astype(np.int64), "predict": preds})
    sample_sub = pd.read_csv(DATA_DIR / "sample_submit.csv")
    final_sub = sample_sub[["event_id"]].merge(sub, on="event_id", how="left").fillna(0)
    final_sub.to_csv("submission_v5_gpu.csv", index=False)
    print("Success! Saved -> submission_v5_gpu.csv")

if __name__ == "__main__":
    train_model()
