import asyncio
import pandas as pd
import time
import os

from original_cell_annotations import run_pipeline

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "table_format.csv"
OUTPUT_FILE = "ragless_new_cellstar_predictions.csv"

BATCH_SIZE = 2
SLEEP_SECONDS = 5

# =====================================================
# LOAD FULL DATASET
# =====================================================

df = pd.read_csv(INPUT_FILE).copy()

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

overall_start = time.perf_counter()

# =====================================================
# MAIN LOOP (BATCH EXECUTION ENGINE)
# =====================================================

for start_row in range(0, total_rows, BATCH_SIZE):

    batch_start = time.perf_counter()

    end_row = min(start_row + BATCH_SIZE, total_rows)

    print("\n========================================")
    print(f"PROCESSING ROWS {start_row} → {end_row - 1}")
    print("========================================\n")

    # Slice batch BEFORE pipeline

    batch_df = df.iloc[start_row:end_row].copy()

    # =================================================
    # Run pipeline
    # =================================================

    try:

        result_df = asyncio.run(run_pipeline(batch_df))

    except Exception as e:

        print(f"[ERROR] Batch {start_row}-{end_row - 1} failed → {e}")
        print("[SKIPPING BATCH]")

        continue

    # =================================================
    # SAVE
    # =================================================

    if first_write:

        result_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        first_write = False

    else:

        result_df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False
        )

    print(f"[SAVED] Batch {start_row}-{end_row - 1}")

    print(
        f"[TIMING] Batch {start_row}-{end_row - 1}: "
        f"{time.perf_counter() - batch_start:.2f} sec"
    )

    # =================================================
    # COOL-DOWN
    # =================================================

    print(f"[SLEEP] {SLEEP_SECONDS}s cooldown...\n")

    time.sleep(SLEEP_SECONDS)

    import gc
    import torch

    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# =====================================================
# COMPLETE
# =====================================================

print("\n========================================")
print("ALL BATCHES COMPLETE")
print(f"FINAL OUTPUT: {OUTPUT_FILE}")
print(
    f"TOTAL TIME: "
    f"{(time.perf_counter() - overall_start)/60:.2f} minutes"
)
print("========================================")