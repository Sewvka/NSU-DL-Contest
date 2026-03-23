import polars as pl
import glob

def create_risk_table():
    print("--- Creating Global MCC Risk Table ---")
    
    # 1. Загружаем лейблы
    labels = pl.read_parquet("data/train_labels.parquet").select([
        pl.col("customer_id"), 
        pl.col("event_id"), 
        pl.col("target")
    ])
    
    # 2. Сканируем тренировочные файлы (только mcc_code)
    train_files = glob.glob("data/train_part_*.parquet")
    
    # Собираем все пары (mcc_code, target)
    dfs = []
    for f in train_files:
        print(f"Reading {f}...")
        df = pl.scan_parquet(f).select(["customer_id", "event_id", "mcc_code"]).collect()
        df = df.join(labels, on=["customer_id", "event_id"], how="inner")
        dfs.append(df.select(["mcc_code", "target"]))
    
    full_df = pl.concat(dfs)
    
    # 3. Считаем средний риск для каждого MCC
    # Добавляем сглаживание (smoothing), чтобы редкие MCC не имели риск 0 или 1
    global_mean = full_df["target"].mean()
    alpha = 100 # вес глобального среднего
    
    risk_table = full_df.group_by("mcc_code").agg([
        pl.count("target").alias("mcc_count"),
        pl.mean("target").alias("mcc_raw_risk")
    ]).with_columns([
        ((pl.col("mcc_raw_risk") * pl.col("mcc_count") + global_mean * alpha) / 
         (pl.col("mcc_count") + alpha)).alias("mcc_global_risk")
    ])
    
    risk_table.select(["mcc_code", "mcc_global_risk"]).write_parquet("features/mcc_risk_table.parquet")
    print(f"Risk table saved. Unique MCCs: {len(risk_table)}")
    print(risk_table.sort("mcc_global_risk", descending=True).head(10))

if __name__ == "__main__":
    import os
    os.makedirs("features", exist_ok=True)
    create_risk_table()
