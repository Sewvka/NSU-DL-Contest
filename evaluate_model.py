import polars as pl
import os
import glob
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve
import gc
from datetime import datetime

def evaluate_model():
    print("--- Detailed Model Evaluation ---")
    
    # 1. Загрузка модели
    if not os.path.exists("baseline_model.cbm"):
        print("Error: baseline_model.cbm not found.")
        return
    model = CatBoostClassifier()
    model.load_model("baseline_model.cbm")
    print("Model loaded.")

    # 2. Подготовка валидационных данных (с 2025-05-01)
    labels = pl.read_parquet("data/train_labels.parquet").select([
        pl.col("customer_id"), 
        pl.col("event_id"), 
        pl.col("target").cast(pl.Int8)
    ])
    stats = pl.read_parquet("features/customer_stats.parquet")
    train_files = sorted(glob.glob("processed/train_*.parquet"))
    
    schema = pl.scan_parquet(train_files[0]).collect_schema()
    drop_cols = ["customer_id", "event_id", "event_dttm", "dttm"]
    features = [c for c in schema.names() if c not in drop_cols]
    
    split_date = datetime(2025, 5, 1)
    val_chunks = []
    
    print("Collecting validation data...")
    for f in train_files:
        # Берем данные только после даты сплита
        q = pl.scan_parquet(f).filter(
            pl.col("event_dttm").str.to_datetime("%Y-%m-%d %H:%M:%S") >= split_date
        )
        
        # Джойним лейблы (все размеченные попадают сюда)
        q = q.join(labels.lazy(), on=["customer_id", "event_id"], how="left")
        
        # Для валидации мы также берем сэмпл неразмеченных (зеленых), 
        # чтобы соотношение классов было похоже на обучающее
        labeled = q.filter(pl.col("target").is_not_null())
        green = q.filter(
            (pl.col("target").is_null()) & 
            (pl.col("event_id").hash(42).mod(100) < 2)
        )
        
        chunk = pl.concat([labeled, green]).with_columns(
            pl.col("target").fill_null(0)
        ).join(stats.lazy(), on="customer_id", how="left").collect()
        
        val_chunks.append(chunk)
        print(f"  Processed {f}, found {chunk.height} val rows")

    val_df = pl.concat(val_chunks)
    print(f"Total Validation size: {val_df.height}")

    y_val = val_df["target"].to_pandas()
    X_cols = features + [c for c in stats.columns if c != "customer_id"]
    X_val = val_df.select(X_cols).to_pandas()
    
    # Авто-определение категорий для корректной обработки
    cat_features = X_val.select_dtypes(exclude=['number']).columns.tolist()
    numeric_cats = ['mcc_code', 'hour', 'weekday', 'day']
    for col in numeric_cats:
        if col in X_val.columns and col not in cat_features:
            cat_features.append(col)
            
    for col in cat_features:
        X_val[col] = X_val[col].astype(str).fillna("NaN")

    # 3. Предсказание
    print("Predicting probabilities...")
    y_pred = model.predict_proba(X_val)[:, 1]
    
    # 4. Метрики
    prauc = average_precision_score(y_val, y_pred)
    print(f"\n>>> FINAL PR-AUC: {prauc:.4f} <<<")
    
    print("\n--- Classification Report at different thresholds ---")
    # Посмотрим, как меняется точность и полнота при разных порогах
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        print(f"\n--- Threshold: {threshold} ---")
        print(classification_report(y_val, y_pred > threshold, digits=4))

    # Вывод топ-10 самых важных признаков
    print("\n--- Top 10 Feature Importance ---")
    import numpy as np
    import pandas as pd
    
    importances = model.get_feature_importance()
    names = model.feature_names_
    
    if names is None:
        names = [f"Feature_{i}" for i in range(len(importances))]
        
    fi = pd.DataFrame({'feature': names, 'importance': importances})
    print(fi.sort_values('importance', ascending=False).head(10))

if __name__ == "__main__":
    evaluate_model()
