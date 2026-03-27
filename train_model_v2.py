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
PROCESSED_DIR = Path("processed_v2")
CACHE_DIR = Path("cache")
RANDOM_SEED = 42
VAL_START = pd.Timestamp("2025-05-01")
RECENT_BORDER = pd.Timestamp("2025-02-01")
NEG_SAMPLE_BORDER_STR = "2025-04-01 00:00:00"

CAT_COLS = [
    "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd",
    "timezone", "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
    "developer_tools_i", "compromised_i",
]

META_COLS = ["event_id", "period", "event_ts", "is_train_sample", "is_test", "train_target_raw", "target_bin", "event_date"]

PRIOR_KEYS = ["event_desc", "mcc_code_i", "timezone", "operating_system_type", "channel_indicator_sub_type", "event_type_nm", "pos_cd"]

def make_weights(raw_target, event_ts):
    # raw_target: 1=red, 0=yellow, -1=green sampled
    w = np.where(raw_target == 1, 10.0, np.where(raw_target == 0, 2.5, 1.0)).astype(np.float32)
    ts = pd.to_datetime(event_ts)
    border = pd.Timestamp(NEG_SAMPLE_BORDER_STR)
    green_mask = (raw_target == -1)
    recent_green = green_mask & (ts >= border)
    old_green = green_mask & (ts < border)
    w[recent_green] = 1.8
    w[old_green] = 3.0
    return w

def train_model():
    print("Loading preprocessed parts...")
    parts = []
    for i in [1, 2, 3]:
        parts.append(pl.read_parquet(PROCESSED_DIR / f"features_part_{i}.parquet"))
    
    features = pl.concat(parts)
    del parts; gc.collect()
    
    print("Joining priors...")
    prior_feature_cols = []
    for key in PRIOR_KEYS:
        prior_df = pl.read_parquet(CACHE_DIR / f"prior_{key}.parquet")
        features = features.join(prior_df, on=key, how="left")
        prior_feature_cols.extend([c for c in prior_df.columns if c != key])
    
    # Fill nulls in priors with mean
    for c in prior_feature_cols:
        mean_val = features[c].mean()
        features = features.with_columns(pl.col(c).fill_null(mean_val))
        
    print(f"Full feature table shape: {features.shape}")
    
    # Split Train/Test
    train_pl = features.filter(pl.col("is_train_sample"))
    test_pl = features.filter(pl.col("is_test"))
    del features; gc.collect()
    
    train_df = train_pl.to_pandas()
    test_df = test_pl.to_pandas()
    del train_pl, test_pl; gc.collect()
    
    train_df["event_ts"] = pd.to_datetime(train_df["event_ts"])
    test_df["event_ts"] = pd.to_datetime(test_df["event_ts"])
    
    feature_cols = [c for c in train_df.columns if c not in META_COLS and c != "target"]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    print(f"Features: {len(feature_cols)} (Cat: {len(cat_cols)}, Num: {len(num_cols)})")
    
    # Preprocessing for CatBoost
    for c in cat_cols:
        train_df[c] = train_df[c].fillna(-1).astype(np.int64)
        test_df[c] = test_df[c].fillna(-1).astype(np.int64)
        
    medians = train_df[num_cols].median()
    train_df[num_cols] = train_df[num_cols].fillna(medians)
    test_df[num_cols] = test_df[num_cols].fillna(medians)
    
    train_df = train_df.sort_values("event_ts").reset_index(drop=True)
    
    # Validation mask
    val_mask = train_df["event_ts"] >= VAL_START
    
    # --- MODEL 1: MAIN ---
    print("\nTraining MAIN model...")
    X_main_tr = train_df.loc[~val_mask, feature_cols]
    y_main_tr = train_df.loc[~val_mask, "target_bin"].astype(np.int8).values
    w_main_tr = make_weights(train_df.loc[~val_mask, "train_target_raw"].values, train_df.loc[~val_mask, "event_ts"].values)
    
    X_main_val = train_df.loc[val_mask, feature_cols]
    y_main_val = train_df.loc[val_mask, "target_bin"].astype(np.int8).values
    w_main_val = make_weights(train_df.loc[val_mask, "train_target_raw"].values, train_df.loc[val_mask, "event_ts"].values)
    
    params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "task_type": "CPU", # Force CPU for memory stability, change to GPU if available
        "early_stopping_rounds": 200,
        "verbose": 100
    }
    
    model_main = CatBoostClassifier(**params)
    model_main.fit(Pool(X_main_tr, y_main_tr, weight=w_main_tr, cat_features=cat_cols), 
                   eval_set=Pool(X_main_val, y_main_val, weight=w_main_val, cat_features=cat_cols))
    
    ap_main = average_precision_score(y_main_val, model_main.predict(X_main_val, prediction_type="RawFormulaVal"))
    print(f"Main Val PR-AUC: {ap_main:.6f}")
    
    # --- MODEL 2: RECENT ---
    print("\nTraining RECENT model...")
    recent_mask = (train_df["event_ts"] >= RECENT_BORDER) | (train_df["train_target_raw"] != -1)
    X_recent_tr = train_df.loc[recent_mask & (~val_mask), feature_cols]
    y_recent_tr = train_df.loc[recent_mask & (~val_mask), "target_bin"].astype(np.int8).values
    w_recent_tr = make_weights(train_df.loc[recent_mask & (~val_mask), "train_target_raw"].values, train_df.loc[recent_mask & (~val_mask), "event_ts"].values)
    
    model_recent = CatBoostClassifier(**params)
    model_recent.fit(Pool(X_recent_tr, y_recent_tr, weight=w_recent_tr, cat_features=cat_cols),
                     eval_set=Pool(X_main_val, y_main_val, weight=w_main_val, cat_features=cat_cols))
    
    ap_recent = average_precision_score(y_main_val, model_recent.predict(X_main_val, prediction_type="RawFormulaVal"))
    print(f"Recent Val PR-AUC: {ap_recent:.6f}")
    
    # Blending
    pred_main_val = model_main.predict(X_main_val, prediction_type="RawFormulaVal")
    pred_recent_val = model_recent.predict(X_main_val, prediction_type="RawFormulaVal")
    blend_val = 0.7 * pred_main_val + 0.3 * pred_recent_val
    ap_blend = average_precision_score(y_main_val, blend_val)
    print(f"Blended Val PR-AUC: {ap_blend:.6f}")
    
    # Final Predictions
    print("\nPredicting on test...")
    X_test = test_df[feature_cols]
    test_pred_main = model_main.predict(X_test, prediction_type="RawFormulaVal")
    test_pred_recent = model_recent.predict(X_test, prediction_type="RawFormulaVal")
    test_pred_blend = 0.7 * test_pred_main + 0.3 * test_pred_recent
    
    sub = pd.DataFrame({
        "event_id": test_df["event_id"].values,
        "predict": test_pred_blend
    })
    
    # Merge with sample submit to ensure all IDs present
    sample_sub = pd.read_csv(DATA_DIR / "sample_submit.csv")
    final_sub = sample_sub[["event_id"]].merge(sub, on="event_id", how="left")
    final_sub["predict"] = final_sub["predict"].fillna(final_sub["predict"].mean())
    
    final_sub.to_csv("submission_v2.csv", index=False)
    print("Saved submission_v2.csv")

if __name__ == "__main__":
    train_model()
