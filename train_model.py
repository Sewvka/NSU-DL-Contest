import polars as pl
import os
import glob
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
import gc
from datetime import datetime

def train_model():
    print("--- Starting Memory-Optimized Training ---")
    
    # 1. Загружаем лейблы и стат-фичи
    labels = pl.read_parquet("data/train_labels.parquet").select([
        pl.col("customer_id"), 
        pl.col("event_id"), 
        pl.col("target").cast(pl.Int8)
    ])
    
    # Статистика клиентов (которую мы собрали ранее)
    stats = pl.read_parquet("features/customer_stats.parquet")
    
    # 2. Определяем колонки
    train_files = sorted(glob.glob("processed/train_*.parquet"))
    # Берем список фичей из первого файла
    schema = pl.scan_parquet(train_files[0]).collect_schema()
    drop_cols = ["customer_id", "event_id", "event_dttm", "dttm"]
    features = [c for c in schema.names() if c not in drop_cols]
    
    # MCC и другие категории
    cat_features = [
        'mcc_code', 'event_type_nm', 'event_desc', 'channel_indicator_type', 
        'channel_indicator_sub_type', 'currency_iso_cd', 'pos_cd', 
        'timezone', 'operating_system_type', 'hour', 'weekday'
    ]
    # Оставляем только те, что есть в данных
    cat_features = [c for c in cat_features if c in features]

    print(f"Features count: {len(features)} + {len(stats.columns)-1} stats")

    # 3. Собираем выборку для обучения (Sampling)
    # Чтобы влезть в 8ГБ, мы возьмем ВСЕ размеченные (87к) и ~1.5 млн неразмеченных (Зеленых)
    full_train_list = []
    
    for f in train_files:
        print(f"Processing {f} for sampling...")
        # Джойним файл с лейблами (Left join чтобы увидеть неразмеченные)
        q = pl.scan_parquet(f).join(labels.lazy(), on=["customer_id", "event_id"], how="left")
        
        # Размеченные (Target 1 или 0 из файла лейблов)
        labeled = q.filter(pl.col("target").is_not_null())
        
        # Неразмеченные (Зеленые) - берем случайные ~2% через хеш event_id
        # Это самый стабильный способ ленивого сэмплирования в Polars
        green = q.filter(
            (pl.col("target").is_null()) & 
            (pl.col("event_id").hash(42).mod(100) < 2)
        )
        
        # Объединяем и заполняем таргет для зеленых (0)
        chunk = pl.concat([labeled, green]).with_columns(
            pl.col("target").fill_null(0)
        ).join(stats.lazy(), on="customer_id", how="left")
        
        full_train_list.append(chunk.collect())
        gc.collect()

    df = pl.concat(full_train_list)
    del full_train_list
    gc.collect()

    # 4. Временной сплит (Валидация на последних данных)
    split_date = datetime(2025, 5, 1)
    train_df = df.filter(pl.col("dttm") < split_date)
    val_df = df.filter(pl.col("dttm") >= split_date)
    
    print(f"Final Train size: {train_df.height}, Val size: {val_df.height}")
    
    # Готовим данные для CatBoost
    y_train = train_df["target"].to_pandas()
    y_val = val_df["target"].to_pandas()
    
    # Все колонки кроме служебных и таргета
    X_cols = features + [c for c in stats.columns if c != "customer_id"]
    X_train = train_df.select(X_cols).to_pandas()
    X_val = val_df.select(X_cols).to_pandas()
    
    del df, train_df, val_df
    gc.collect()

    # Обработка категорий (строки для CatBoost)
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str).fillna("NaN")
            X_val[col] = X_val[col].astype(str).fillna("NaN")

    # 5. Обучение
    print("Training CatBoost...")
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        eval_metric='PRAUC', # Целевая метрика соревнования
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
        task_type="CPU"
    )
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model.fit(train_pool, eval_set=val_pool)
    
    # 6. Валидация
    y_pred = model.predict_proba(val_pool)[:, 1]
    prauc = average_precision_score(y_val, y_pred)
    print(f"\n--- Validation PR-AUC: {prauc:.4f} ---")
    
    model.save_model("baseline_model.cbm")
    print("Model saved to baseline_model.cbm")

if __name__ == "__main__":
    train_model()
