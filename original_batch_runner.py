import asyncio
import pandas as pd
import time
import os

from cell_annotations import run_pipeline

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "cellstar_preprocessed.csv"
OUTPUT_FILE = "cellstar_predictions.csv"

BATCH_SIZE = 2
SLEEP_SECONDS = 5

# =====================================================
# LOAD FULL DATASET
# =====================================================

df = pd.read_csv(INPUT_FILE)
df = df.iloc[[1]].copy()

total_rows = len(df)

print(f"[INFO] Total rows in dataset: {total_rows}")
print(f"[INFO] Batch size: {BATCH_SIZE}")
print(f"[INFO] Total batches: {(total_rows + BATCH_SIZE - 1) // BATCH_SIZE}")

# =====================================================
# RESET OUTPUT FILE
# =====================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

first_write = True

# =====================================================
# MAIN LOOP (BATCH EXECUTION ENGINE)
# =====================================================

for start_row in range(0, total_rows, BATCH_SIZE):

    end_row = min(start_row + BATCH_SIZE, total_rows)

    print("\n========================================")
    print(f"PROCESSING ROWS {start_row} → {end_row - 1}")
    print("========================================\n")

    # Slice batch BEFORE pipeline
    batch_df = df.iloc[start_row:end_row].copy()

    # Run pipeline on ONLY this batch
    try:
        result_df = asyncio.run(run_pipeline(batch_df))

    except Exception as e:
        print(f"[ERROR] Batch {start_row}-{end_row - 1} failed → {e}")
        print("[SKIPPING BATCH]")
        continue

    # =================================================
    # SAFE APPEND TO OUTPUT
    # =================================================

    if first_write:
        result_df.to_csv(OUTPUT_FILE, index=False)
        first_write = False
    else:
        result_df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False
        )

    print(f"[SAVED] Batch {start_row}-{end_row - 1}")

    # =================================================
    # COOL-DOWN (MPS / MEMORY STABILITY)
    # =================================================

    print(f"[SLEEP] {SLEEP_SECONDS}s cooldown...\n")
    time.sleep(SLEEP_SECONDS)
    import gc
    import torch

    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

print("\n========================================")
print("ALL BATCHES COMPLETE")
print(f"FINAL OUTPUT: {OUTPUT_FILE}")
print("========================================")