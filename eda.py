import polars as pl
import os

def run_eda():
    print("--- Target Distribution ---")
    if os.path.exists("data/train_labels.parquet"):
        labels = pl.read_parquet("data/train_labels.parquet")
        print(f"Total labeled rows: {len(labels)}")
        print(labels["target"].value_counts())
        
        # Missing values in target
        null_count = labels["target"].is_null().sum()
        print(f"Null target count: {null_count}")
    else:
        print("data/train_labels.parquet not found.")

    print("\n--- Feature Structure (train_part_1) ---")
    if os.path.exists("data/train_part_1.parquet"):
        # Reading only first 5 rows to save memory/time
        train_sample = pl.read_parquet("data/train_part_1.parquet", n_rows=5)
        print(f"Columns ({len(train_sample.columns)} total):")
        print(train_sample.columns)
        print("\nFirst row sample:")
        print(train_sample.head(1))
    else:
        print("data/train_part_1.parquet not found.")

    print("\n--- Test Structure ---")
    if os.path.exists("data/test.parquet"):
        test_sample = pl.read_parquet("data/test.parquet", n_rows=5)
        print(f"Test Columns ({len(test_sample.columns)} total):")
        print(test_sample.columns)
    else:
        print("data/test.parquet not found.")

if __name__ == "__main__":
    run_eda()
