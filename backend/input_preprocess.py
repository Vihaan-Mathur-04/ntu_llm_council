import pandas as pd

INPUT_PATH = "/Users/vihaan_mathur/ntu_project/llm_council/2024-09-01_cellstar.csv"
OUTPUT_PATH = "/Users/vihaan_mathur/ntu_project/llm_council/cellstar_preprocessed.csv"

CELL_ID_COL = "Cell_ID"
CELL_NAME_COL = "Cell_Name"
MARKER_COL = "Marker"


def load_data():

    df = pd.read_csv(INPUT_PATH)

    print(f"[INFO] Loaded shape: {df.shape}")

    return df


def aggregate_cells(df):

    grouped_rows = []

    grouped = df.groupby(CELL_ID_COL, sort=False)

    for cell_id, group in grouped:

        # keep first cell name encountered
        cell_name = str(group[CELL_NAME_COL].iloc[0])

        # IMPORTANT:
        # keep EVERY marker occurrence
        markers = (
            group[MARKER_COL]
            .fillna("")
            .astype(str)
            .tolist()
        )

        markers = [
            m.strip()
            for m in markers
            if m.strip()
        ]

        grouped_rows.append({
            "Cell_ID": cell_id,
            "Cell_Name": cell_name,
            "Marker_Genes": ",".join(markers),
            "Marker_Count": len(markers)
        })

    return pd.DataFrame(grouped_rows)


def main():

    print("Loading data...")

    df = load_data()

    print("Grouping by Cell_ID...")

    result = aggregate_cells(df)

    print(f"[INFO] Output shape: {result.shape}")

    print("\n[MARKER STATS]")
    print(result["Marker_Count"].describe())

    print("\n[TOP 10 LARGEST CELLS]")
    print(
        result.sort_values(
            "Marker_Count",
            ascending=False
        )[
            ["Cell_ID", "Cell_Name", "Marker_Count"]
        ].head(10)
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\n[SAVED] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()