"""
CellStar EDA (Critical sanity check before RAG / council pipeline)
"""

import pandas as pd
import numpy as np

INPUT_PATH = "/Users/vihaan_mathur/ntu_project/llm_council/cellstar_preprocessed.csv"

CELL_TYPE_COL = "Cell_ID"
MARKER_COL = "Marker_Genes"
COUNT_COL = "Marker_Count"


def load_data():

    df = pd.read_csv(INPUT_PATH)

    print(f"[INFO] Loaded shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    return df


def compute_stats(df):

    # --------------------------------------------------
    # CELL TYPES
    # --------------------------------------------------

    if CELL_TYPE_COL in df.columns:

        num_cell_types = df[CELL_TYPE_COL].nunique()

        print(f"\n[STATS] Unique cell types: {num_cell_types}")

        print("\n[TOP CELL TYPES]")
        print(df[CELL_TYPE_COL].value_counts().head(10))

    else:

        print("[WARN] Cell type column not found")

    # --------------------------------------------------
    # MARKER COUNTS
    # --------------------------------------------------

    if COUNT_COL not in df.columns:

        df[COUNT_COL] = df[MARKER_COL].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )

    counts = df[COUNT_COL].values

    print("\n[MARKER STATS]")
    print(f"Mean markers per cell : {np.mean(counts):.2f}")
    print(f"Std dev               : {np.std(counts):.2f}")
    print(f"Median                : {np.median(counts)}")
    print(f"Min                   : {np.min(counts)}")
    print(f"Max                   : {np.max(counts)}")

    # --------------------------------------------------
    # GLOBAL MARKER STATISTICS
    # --------------------------------------------------

    total_marker_entries = 0
    unique_markers = set()

    for marker_string in df[MARKER_COL]:

        if pd.isna(marker_string):
            continue

        markers = [
            m.strip().upper()
            for m in str(marker_string).split(",")
            if m.strip()
        ]

        total_marker_entries += len(markers)

        unique_markers.update(markers)

    duplicate_marker_entries = (
        total_marker_entries
        - len(unique_markers)
    )

    print("\n[GLOBAL MARKER STATISTICS]")
    print(f"Total marker entries : {total_marker_entries}")
    print(f"Unique marker names  : {len(unique_markers)}")
    print(f"Duplicate entries    : {duplicate_marker_entries}")

    # --------------------------------------------------
    # LARGE CELLS
    # --------------------------------------------------

    over_1000 = np.sum(counts > 1000)

    print(f"\nCells with >1000 markers : {over_1000}")

    # --------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------

    q99 = np.percentile(counts, 99)

    outliers = df[df[COUNT_COL] > q99]

    print("\n[OUTLIERS]")
    print(f"99th percentile cutoff : {q99:.2f}")
    print(f"Number of extreme rows : {len(outliers)}")

    if len(outliers) > 0:

        print("\nExample extreme rows:")

        print(
            outliers[
                [CELL_TYPE_COL, COUNT_COL]
            ].head()
        )


def main():

    df = load_data()

    compute_stats(df)


if __name__ == "__main__":

    main()