import polars as pl
import pandas as pd
import os
import glob
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
import gc
from datetime import datetime

def train_model():
    print("--- Starting FINAL Elite Training on 100% Data ---")
    
    # 1. Загружаем лейблы и "чистую" базу статистики (без утечек из будущего)
    labels = pl.read_parquet("data/train_labels.parquet").select([
        pl.col("customer_id"), 
        pl.col("event_id"), 
        pl.col("target").cast(pl.Int8)
    ])
    stats = pl.read_parquet("features/train_baseline_stats.parquet")
    
    # 2. Список файлов и колонок
    train_files = sorted(glob.glob("processed/train_*.parquet"))
    schema = pl.scan_parquet(train_files[0]).collect_schema()
    
    drop_cols = ["customer_id", "event_id", "event_dttm", "dttm", "session_id", "device_system_version"]
    features = [c for c in schema.names() if c not in drop_cols]
    
    # ВАЛИДАЦИЯ: Последние 10 дней мая 2025 для проверки
    split_date = datetime(2025, 5, 20)
    print(f"Validation Split Date: {split_date}")

    # Полный список признаков (базовые + из статистики + вычисляемые на лету)
    new_cols = [
        "amt_to_avg_ratio", "amt_to_max_ratio", "is_night", 
        "sec_since_last_trans", "user_mcc_count", "mcc_amt_share", "is_new_mcc",
        "trans_count_10m", "trans_count_1h", "avg_amt_1h",
        "mcc_global_risk", "amt_z_score_mcc", "amt_to_recent_avg", "is_mcc_change"
    ]
    
    final_cols = list(set(features + [c for c in stats.columns if c != "customer_id"] + new_cols))
    X_cols = [c for c in final_cols if c not in drop_cols]
    
    # 3. Подготовка валидации ( sampled for memory )
    print("Preparing Validation Set (Sampled to 500k rows)...")
    X_val_list, y_val_list = [], []
    cat_features = None

    for f in train_files:
        q = pl.scan_parquet(f).filter(
            (pl.col("dttm") >= split_date) & 
            (pl.col("event_id").hash(42).mod(100) < 5)
        ).join(labels.lazy(), on=["customer_id", "event_id"], how="left"
        ).with_columns(pl.col("target").fill_null(0)
        ).join(stats.lazy(), on="customer_id", how="left").with_columns([
            (pl.col("operaton_amt") / (pl.col("user_avg_amt") + 1)).cast(pl.Float32).alias("amt_to_avg_ratio"),
            (pl.col("operaton_amt") / (pl.col("user_max_amt") + 1)).cast(pl.Float32).alias("amt_to_max_ratio"),
            pl.when(pl.col("hour").is_between(0, 6)).then(1).otherwise(0).cast(pl.Int8).alias("is_night")
        ])
        
        df_v = q.collect(engine="streaming")
        if df_v.height > 0:
            actual_cols = [c for c in X_cols if c in df_v.columns]
            X_v_chunk = df_v.select(actual_cols).to_pandas()
            if cat_features is None:
                cat_features = X_v_chunk.select_dtypes(exclude=['number']).columns.tolist()
                numeric_cats = ['mcc_code', 'hour', 'weekday', 'day', 'is_night', 'is_mcc_change', 'is_new_mcc']
                cat_features = sorted(list(set(cat_features + [c for c in numeric_cats if c in X_v_chunk.columns])))
            
            for col in cat_features:
                X_v_chunk[col] = X_v_chunk[col].astype(str).fillna("NaN")
            X_val_list.append(X_v_chunk)
            y_val_list.append(df_v["target"].to_pandas())
        del df_v; gc.collect()

    X_val = pd.concat(X_val_list); del X_val_list
    y_val = pd.concat(y_val_list); del y_val_list
    print(f"Validation data ready: {len(X_val)} rows.")

    # 4. Итеративное обучение
    model_path = "current_model.cbm"
    if os.path.exists(model_path): os.remove(model_path)
    
    pos_weight = 300
    params = {
        'iterations': 250, # Увеличено для максимального качества
        'learning_rate': 0.04,
        'depth': 7,        # Оптимально для 16ГБ
        'eval_metric': 'PRAUC',
        'random_seed': 42,
        'verbose': 50,
        'scale_pos_weight': pos_weight,
        'task_type': "CPU"
    }

    step = 0
    for f in train_files:
        print(f"\n--- Processing File: {f} ---")
        total_rows = pl.scan_parquet(f).filter(pl.col("dttm") < split_date).collect().height
        chunk_size = 5_000_000
        
        for offset in range(0, total_rows, chunk_size):
            step += 1
            print(f"  Step {step}: rows {offset} to {min(offset+chunk_size, total_rows)}...")
            
            q = pl.scan_parquet(f).filter(pl.col("dttm") < split_date).slice(offset, chunk_size).join(
                labels.lazy(), on=["customer_id", "event_id"], how="left"
            ).with_columns(pl.col("target").fill_null(0)
            ).join(stats.lazy(), on="customer_id", how="left").with_columns([
                (pl.col("operaton_amt") / (pl.col("user_avg_amt") + 1)).cast(pl.Float32).alias("amt_to_avg_ratio"),
                (pl.col("operaton_amt") / (pl.col("user_max_amt") + 1)).cast(pl.Float32).alias("amt_to_max_ratio"),
                pl.when(pl.col("hour").is_between(0, 6)).then(1).otherwise(0).cast(pl.Int8).alias("is_night")
            ])
            
            df_t = q.collect(engine="streaming")
            if df_t.height == 0: continue
                
            actual_cols = [c for c in X_cols if c in df_t.columns]
            X_t = df_t.select(actual_cols).to_pandas()
            for col in cat_features:
                X_t[col] = X_t[col].astype(str).fillna("NaN")
            y_t = df_t["target"].to_pandas()
            
            if y_t.nunique() < 2:
                print(f"  Skipping Step {step}: target contains only one class.")
                del df_t, X_t, y_t; gc.collect()
                continue

            train_pool = Pool(X_t, y_t, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            del df_t, X_t, y_t; gc.collect()

            current_params = params.copy()
            if step > 1:
                current_params['learning_rate'] = 0.02 

            clf = CatBoostClassifier(**current_params)
            
            if step == 1:
                clf.fit(train_pool, eval_set=val_pool)
            else:
                clf.fit(train_pool, eval_set=val_pool, init_model=model_path)
            
            clf.save_model(model_path)
            del train_pool, val_pool, clf; gc.collect()

    # 5. Итоговые результаты и важность признаков
    final_model = CatBoostClassifier().load_model(model_path)
    final_val_pool = Pool(X_val, y_val, cat_features=cat_features)
    y_pred = final_model.predict_proba(final_val_pool)[:, 1]
    
    prauc = average_precision_score(y_val, y_pred)
    rocauc = roc_auc_score(y_val, y_pred)
    
    print(f"\n--- FINAL ELITE RESULTS ---")
    print(f"PR-AUC:  {prauc:.6f}")
    print(f"ROC-AUC: {rocauc:.6f}")
    
    # Вывод важности признаков
    feat_imp = pd.DataFrame({
        'feature': final_model.feature_names_,
        'importance': final_model.get_feature_importance()
    }).sort_values(by='importance', ascending=False)
    
    print("\nTop 20 Important Features:")
    print(feat_imp.head(20))
    
    if os.path.exists("baseline_model.cbm"):
        os.remove("baseline_model.cbm")
    os.rename(model_path, "baseline_model.cbm")
    print("Final model saved to baseline_model.cbm")

if __name__ == "__main__":
    train_model()
