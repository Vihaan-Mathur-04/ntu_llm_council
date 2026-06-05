"""
CellStar EDA (Critical sanity check before RAG / council pipeline)
"""

import pandas as pd
import numpy as np

INPUT_PATH = "/Users/vihaan_mathur/ntu_project/llm_council/2024-09-01_cellstar.csv"

CELL_TYPE_COL = "Cell_ID"        # adjust if needed
MARKER_COL = "Marker"
COUNT_COL = "Marker_Count"


def load_data():
    df = pd.read_csv(INPUT_PATH)

    print(f"[INFO] Loaded shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    return df


def compute_stats(df):

    # -----------------------------
    # CELL TYPE ANALYSIS
    # -----------------------------
    if CELL_TYPE_COL in df.columns:
        num_cell_types = df[CELL_TYPE_COL].nunique()
        print(f"\n[STATS] Unique cell types: {num_cell_types}")

        print("\n[TOP CELL TYPES]")
        print(df[CELL_TYPE_COL].value_counts().head(10))

    else:
        print("[WARN] Cell type column not found")

    # -----------------------------
    # MARKER COUNT ANALYSIS
    # -----------------------------
    if COUNT_COL not in df.columns:
        # fallback compute
        df[COUNT_COL] = df[MARKER_COL].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )

    counts = df[COUNT_COL].values

    print("\n[MARKER STATS]")
    print(f"Mean markers per cell: {np.mean(counts):.2f}")
    print(f"Std dev: {np.std(counts):.2f}")
    print(f"Min: {np.min(counts)}")
    print(f"Max: {np.max(counts)}")
    print(f"Median: {np.median(counts)}")

    # -----------------------------
    # OUTLIER DETECTION
    # -----------------------------
    q99 = np.percentile(counts, 99)

    outliers = df[counts > q99]

    print(f"\n[OUTLIERS]")
    print(f"99th percentile cutoff: {q99}")
    print(f"Number of extreme cells: {len(outliers)}")

    if len(outliers) > 0:
        print("\nExample extreme rows:")
        print(outliers[[CELL_TYPE_COL, COUNT_COL]].head())


def main():
    df = load_data()
    compute_stats(df)


if __name__ == "__main__":
    main()