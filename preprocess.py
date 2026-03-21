import polars as pl
import os
import glob

def preprocess_data(input_pattern, output_prefix, tech_cols):
    files = sorted(glob.glob(input_pattern))
    total_files = len(files)
    print(f"Found {total_files} files for {output_prefix}")
    
    for i, file_path in enumerate(files):
        output_path = f"{output_prefix}_{i}.parquet"
        
        # Checkpoint: skip if output file already exists
        if os.path.exists(output_path):
            print(f"[{i+1}/{total_files}] Skipping {file_path} (already processed)")
            continue
            
        print(f"[{i+1}/{total_files}] Processing {file_path}...")
        
        # 1. Use Lazy API (scan_parquet)
        q = pl.scan_parquet(file_path)
        schema = q.collect_schema()
        
        # 2. Temporal Features
        q = q.with_columns([
            pl.col("event_dttm").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("dttm")
        ])
        q = q.with_columns([
            pl.col("dttm").dt.hour().cast(pl.Int8).alias("hour"),
            pl.col("dttm").dt.weekday().cast(pl.Int8).alias("weekday"),
            pl.col("dttm").dt.day().cast(pl.Int8).alias("day")
        ])
        
        # 3. Fill nulls and cast technical flags to Int8
        tech_ops = []
        for col in tech_cols:
            if col in schema:
                col_expr = pl.col(col)
                if schema[col] == pl.String:
                    col_expr = col_expr.cast(pl.Int32, strict=False)
                
                tech_ops.append(
                    col_expr.fill_null(-1).cast(pl.Int8).alias(col)
                )
        
        # Additional optimizations: downcast common columns
        if "operaton_amt" in schema:
            tech_ops.append(pl.col("operaton_amt").cast(pl.Float32))
        if "mcc_code" in schema:
            tech_ops.append(pl.col("mcc_code").cast(pl.Int32, strict=False))
            
        if tech_ops:
            q = q.with_columns(tech_ops)

        # 4. Execute with Streaming to save memory
        try:
            # sink_parquet is even more memory-efficient than collect().write_parquet()
            # but it has some limitations. Streaming collect is safer here.
            df = q.collect(streaming=True)
            df.write_parquet(output_path)
            print(f"   Successfully saved to {output_path}")
        except Exception as e:
            print(f"   Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    os.makedirs("processed", exist_ok=True)
    
    tech_columns = ["developer_tools", "phone_voip_call_state", "web_rdp_connection", "compromised"]
    
    print("--- Preprocessing train parts ---")
    preprocess_data("data/train_part_*.parquet", "processed/train", tech_columns)
    
    print("\n--- Preprocessing test ---")
    preprocess_data("data/test.parquet", "processed/test", tech_columns)
    
    print("\n--- Preprocessing pretrain parts (optional) ---")
    preprocess_data("data/pretrain_part_*.parquet", "processed/pretrain", tech_columns)
    
    print("\nDone.")
