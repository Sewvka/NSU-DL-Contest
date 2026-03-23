import polars as pl
import pandas as pd
import glob
from catboost import CatBoostClassifier, Pool
import gc

def make_submission():
    print("--- Preparing Submission ---")
    
    # 1. Загружаем модель
    model = CatBoostClassifier().load_model("baseline_model.cbm")
    features_in_model = model.feature_names_
    print(f"Model loaded. Expected features: {len(features_in_model)}")
    
    # 2. Загружаем статистику клиентов
    stats = pl.read_parquet("features/customer_stats.parquet")
    
    # 3. Обрабатываем тестовый файл
    # В этой задаче тестовый файл обычно один (test_0.parquet)
    test_files = glob.glob("processed/test_*.parquet")
    
    all_preds = []
    
    for f in test_files:
        print(f"Processing {f}...")
        df = pl.read_parquet(f)
        
        # Важно: Признаки в тесте должны считаться ТАК ЖЕ, как в трейне
        # (Они уже частично посчитаны в preprocess.py, добавим недостающие)
        df = df.join(stats, on="customer_id", how="left")
        
        # 3. Досчитываем динамические признаки (те же, что в preprocess.py)
        df = df.sort(["customer_id", "dttm"])
        
        # Временная колонка для подсчета
        df = df.with_columns(pl.lit(1).alias("_ones"))
        
        # Rolling & Profiling (Копируем логику из preprocess.py)
        df = df.with_columns([
            pl.col("_ones").rolling_sum_by("dttm", window_size="10m").over("customer_id").alias("trans_count_10m"),
            pl.col("_ones").rolling_sum_by("dttm", window_size="1h").over("customer_id").alias("trans_count_1h"),
            pl.col("operaton_amt").rolling_mean_by("dttm", window_size="1h").over("customer_id").alias("avg_amt_1h"),
            
            (pl.col("dttm").diff().dt.total_seconds().over("customer_id")).fill_null(999999).alias("sec_since_last_trans"),
            pl.col("operaton_amt").mean().over("customer_id").alias("user_avg_amt_all"),
            pl.col("operaton_amt").std().over("customer_id").fill_null(0).alias("user_std_amt_all"),
            
            pl.col("mcc_code").cum_count().over(["customer_id", "mcc_code"]).alias("user_mcc_count"),
            pl.col("operaton_amt").cum_sum().over(["customer_id", "mcc_code"]).alias("user_mcc_cum_amt"),
            pl.col("operaton_amt").cum_sum().over("customer_id").alias("user_total_cum_amt")
        ])
        
        df = df.with_columns([
            ((pl.col("operaton_amt") - pl.col("user_avg_amt_all")) / (pl.col("user_std_amt_all") + 1)).alias("amt_z_score"),
            (pl.col("operaton_amt") / (pl.col("avg_amt_1h") + 1)).alias("amt_to_recent_avg"),
            (pl.col("user_mcc_cum_amt") / pl.col("user_mcc_count")).alias("user_mcc_avg_amt"),
            (pl.col("user_mcc_cum_amt") / (pl.col("user_total_cum_amt") + 1)).alias("mcc_amt_share"),
            (pl.col("user_mcc_count") == 1).cast(pl.Int8).alias("is_new_mcc"),
            
            # Старые признаки (для совместимости)
            (pl.col("operaton_amt") / (pl.col("user_avg_amt") + 1)).alias("amt_to_avg_ratio"),
            (pl.col("operaton_amt") / (pl.col("user_max_amt") + 1)).alias("amt_to_max_ratio"),
            pl.when(pl.col("hour").is_between(0, 6)).then(1).otherwise(0).cast(pl.Int8).alias("is_night")
        ])
        
        # Отбираем колонки, которые есть в модели
        available_cols = [c for c in features_in_model if c in df.columns]
        X_test = df.select(available_cols).to_pandas()
        event_ids = df["event_id"].to_pandas()
        
        # Обработка категориальных признаков (приводим к строкам как в трейне)
        cat_features = X_test.select_dtypes(exclude=['number']).columns.tolist()
        numeric_cats = ['mcc_code', 'hour', 'weekday', 'day', 'is_night']
        cat_features = list(set(cat_features + [c for c in numeric_cats if c in X_test.columns]))
        
        for col in cat_features:
            X_test[col] = X_test[col].astype(str).fillna("NaN")
            
        print(f"Predicting on {len(X_test)} rows...")
        preds = model.predict_proba(X_test)[:, 1]
        
        res = pd.DataFrame({
            "event_id": event_ids,
            "target": preds
        })
        all_preds.append(res)
        
        del df, X_test, res; gc.collect()
        
    # 4. Сохраняем результат
    final_sub = pd.concat(all_preds)
    final_sub.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved to submission.csv ({len(final_sub)} rows)")
    print("Top 5 predictions:")
    print(final_sub.head())

if __name__ == "__main__":
    make_submission()
