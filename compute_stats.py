import polars as pl
import glob
import os

def compute_user_stats():
    print("Computing global user statistics with memory optimizations...")
    
    patterns = [
        "data/train_part_*.parquet",
        "data/test.parquet",
        "data/pretrain_part_*.parquet"
    ]
    
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p))
    
    print(f"Aggregating from {len(all_files)} files...")
    
    # Сразу переводим в компактные типы
    q = pl.scan_parquet(all_files).select([
        pl.col("customer_id"),
        pl.col("operaton_amt").cast(pl.Float32),
        pl.col("mcc_code").cast(pl.Int32, strict=False),
        # Конвертация в Datetime (8 байт) ДО агрегации экономит много RAM по сравнению со строками
        pl.col("event_dttm").str.to_datetime("%Y-%m-%d %H:%M:%S")
    ])
    
    stats = q.group_by("customer_id").agg([
        pl.len().alias("user_event_count"),
        pl.col("operaton_amt").mean().alias("user_avg_amt"),
        pl.col("operaton_amt").std().alias("user_std_amt"),
        pl.col("operaton_amt").max().alias("user_max_amt"),
        pl.col("mcc_code").n_unique().alias("user_unique_mcc_count"),
        pl.col("event_dttm").min().alias("min_dttm"),
        pl.col("event_dttm").max().alias("max_dttm"),
    ])
    
    print("Collecting stats (streaming mode enabled)...")
    df_stats = stats.collect(streaming=True)
    
    print("Post-processing...")
    df_stats = df_stats.with_columns([
        ((pl.col("max_dttm") - pl.col("min_dttm")).dt.total_days()).alias("user_days_active")
    ]).drop(["min_dttm", "max_dttm"])
    
    # Финальное сжатие результатов
    df_stats = df_stats.with_columns([
        pl.col("user_event_count").cast(pl.Int32),
        pl.col("user_unique_mcc_count").cast(pl.Int16),
        pl.col("user_days_active").cast(pl.Int16),
    ])
    
    os.makedirs("features", exist_ok=True)
    df_stats.write_parquet("features/customer_stats.parquet")
    print(f"Stats saved. Memory safe. Users: {df_stats.height}")

if __name__ == "__main__":
    compute_user_stats()
