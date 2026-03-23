import polars as pl
import os
import glob
import gc

def preprocess_data(input_pattern, output_prefix, tech_cols):
    files = sorted(glob.glob(input_pattern))
    total_files = len(files)
    print(f"Found {total_files} files for {output_prefix}")
    
    # 0. Загружаем таблицу глобальных рисков MCC
    risk_table = pl.read_parquet("features/mcc_risk_table.parquet")
    print(f"Risk table loaded. Average global risk: {risk_table['mcc_global_risk'].mean():.6f}")
    
    for i, file_path in enumerate(files):
        output_path = f"{output_prefix}_{i}.parquet"
        print(f"[{i+1}/{total_files}] Processing {file_path} (Elite Features Mode)...")
        
        # 1. Загрузка и базовая подготовка
        df = pl.read_parquet(file_path)
        df = df.with_columns([
            pl.col("event_dttm").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("dttm"),
            pl.lit(1).alias("_ones")
        ])
        
        # Сортировка по времени внутри пользователя
        df = df.sort(["customer_id", "dttm"])
        
        # 2. Временные признаки
        df = df.with_columns([
            pl.col("dttm").dt.hour().cast(pl.Int8).alias("hour"),
            pl.col("dttm").dt.weekday().cast(pl.Int8).alias("weekday")
        ])
        
        # 3. Глобальные риски (Target Encoding)
        df = df.join(risk_table, on="mcc_code", how="left").with_columns(
            pl.col("mcc_global_risk").fill_null(risk_table["mcc_global_risk"].mean())
        )
        
        # 4. Локальные аномалии внутри категорий (Z-Score MCC)
        df = df.with_columns([
            # Среднее и стандартное отклонение клиента ВНУТРИ ЭТОГО MCC
            pl.col("operaton_amt").mean().over(["customer_id", "mcc_code"]).alias("user_mcc_avg_amt_all"),
            pl.col("operaton_amt").std().over(["customer_id", "mcc_code"]).fill_null(0).alias("user_mcc_std_amt_all"),
            
            # Флаг смены MCC
            (pl.col("mcc_code") != pl.col("mcc_code").shift(1).over("customer_id")).cast(pl.Int8).fill_null(1).alias("is_mcc_change")
        ])
        
        # 5. Velocity & Anomaly Metrics
        df = df.with_columns([
            # Динамические окна 10м и 1ч
            pl.col("_ones").rolling_sum_by("dttm", window_size="10m").over("customer_id").alias("trans_count_10m"),
            pl.col("_ones").rolling_sum_by("dttm", window_size="1h").over("customer_id").alias("trans_count_1h"),
            pl.col("operaton_amt").rolling_mean_by("dttm", window_size="1h").over("customer_id").alias("avg_amt_1h"),
            
            # Временные интервалы
            (pl.col("dttm").diff().dt.total_seconds().over("customer_id")).fill_null(999999).alias("sec_since_last_trans"),
            
            # Общие профили
            pl.col("mcc_code").cum_count().over(["customer_id", "mcc_code"]).alias("user_mcc_count"),
            pl.col("operaton_amt").cum_sum().over(["customer_id", "mcc_code"]).alias("user_mcc_cum_amt"),
            pl.col("operaton_amt").cum_sum().over("customer_id").alias("user_total_cum_amt")
        ])
        
        # 6. Финальный расчет аномалий
        df = df.with_columns([
            # Насколько сумма аномальна ДЛЯ ЭТОГО MCC для этого клиента
            ((pl.col("operaton_amt") - pl.col("user_mcc_avg_amt_all")) / (pl.col("user_mcc_std_amt_all") + 1)).alias("amt_z_score_mcc"),
            
            # Общая доля суммы
            (pl.col("user_mcc_cum_amt") / (pl.col("user_total_cum_amt") + 1)).alias("mcc_amt_share"),
            
            # Отношение суммы к локальному среднему за час
            (pl.col("operaton_amt") / (pl.col("avg_amt_1h") + 1)).alias("amt_to_recent_avg"),
            
            # Флаг первой транзакции в новом MCC
            (pl.col("user_mcc_count") == 1).cast(pl.Int8).alias("is_new_mcc")
        ])

        # 7. Очистка типов
        schema = df.schema
        tech_ops = []
        for col in tech_cols:
            if col in schema:
                if schema[col] == pl.String:
                    tech_ops.append(pl.col(col).cast(pl.Int32, strict=False).fill_null(-1).cast(pl.Int8))
                else:
                    tech_ops.append(pl.col(col).fill_null(-1).cast(pl.Int8))
        
        tech_ops.append(pl.col("operaton_amt").cast(pl.Float32))
        tech_ops.append(pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1))
        
        # Все аномальные признаки в Float32
        float_cols = [
            "mcc_global_risk", "amt_z_score_mcc", "mcc_amt_share", 
            "sec_since_last_trans", "amt_to_recent_avg", "avg_amt_1h",
            "trans_count_10m", "trans_count_1h"
        ]
        for c in float_cols:
            if c in df.columns:
                tech_ops.append(pl.col(c).cast(pl.Float32))
        
        # Удаляем временные данные
        temp_cols = [
            "_ones", "user_mcc_avg_amt_all", "user_mcc_std_amt_all", 
            "user_mcc_cum_amt", "user_total_cum_amt"
        ]
        df = df.with_columns(tech_ops).drop(temp_cols)
        
        df.write_parquet(output_path)
        print(f"   Saved to {output_path}")
        
        del df; gc.collect()

if __name__ == "__main__":
    os.makedirs("processed", exist_ok=True)
    tech_columns = ["developer_tools", "phone_voip_call_state", "web_rdp_connection", "compromised"]
    
    print("--- Preprocessing train parts (Elite Anomaly Features) ---")
    preprocess_data("data/train_part_*.parquet", "processed/train", tech_columns)
    
    print("\n--- Preprocessing test ---")
    preprocess_data("data/test.parquet", "processed/test", tech_columns)
    
    print("\nDone.")
