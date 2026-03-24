import polars as pl
import glob
import os

def compute_group_stats(files, output_path):
    print(f"Aggregating from {len(files)} files into {output_path}...")
    
    q = pl.scan_parquet(files).select([
        pl.col("customer_id"),
        pl.col("operaton_amt").cast(pl.Float32),
        pl.col("mcc_code").cast(pl.Int32, strict=False),
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
    
    # Сжатие типов
    df_stats = df_stats.with_columns([
        pl.col("user_event_count").cast(pl.Int32),
        pl.col("user_unique_mcc_count").cast(pl.Int16),
        pl.col("user_days_active").cast(pl.Int16),
    ])
    
    df_stats.write_parquet(output_path)
    print(f"Done. Users: {df_stats.height}")

def main():
    os.makedirs("features", exist_ok=True)
    
    pretrain_files = glob.glob("data/pretrain_part_*.parquet")
    train_files = glob.glob("data/train_part_*.parquet")
    
    # 1. База для обучения (только Pre-train)
    print("\n--- Computing Train Baseline Stats (Pre-train only) ---")
    compute_group_stats(pretrain_files, "features/train_baseline_stats.parquet")
    
    # 2. База для теста (Pre-train + Train)
    print("\n--- Computing Test Baseline Stats (Pre-train + Train) ---")
    compute_group_stats(pretrain_files + train_files, "features/test_baseline_stats.parquet")
    
    # Для обратной совместимости (пока не обновим остальные скрипты)
    # Но лучше использовать раздельные файлы
    print("\nSyncing legacy customer_stats.parquet with test baseline...")
    pl.read_parquet("features/test_baseline_stats.parquet").write_parquet("features/customer_stats.parquet")

if __name__ == "__main__":
    main()
