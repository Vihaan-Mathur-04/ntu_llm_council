import random
import pandas as pd

INPUT_PATH = "/Users/vihaan_mathur/ntu_project/llm_council/cellstar_reference_preprocessed.csv"
OUTPUT_PATH = "/Users/vihaan_mathur/ntu_project/llm_council/cellstar_chunked_reference.csv"

THRESHOLD = 285
WINDOW = 200
OVERLAP = 50
STEP = WINDOW - OVERLAP

# Fixed seed for reproducibility
RANDOM_SEED = 42


def chunk_markers(markers):

    chunks = []

    start = 0
    n = len(markers)

    while True:

        if n - start <= 250:
            chunks.append(markers[start:])
            break

        end = start + WINDOW

        chunks.append(markers[start:end])

        start += STEP

    return chunks


def main():

    random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_PATH)

    new_rows = []

    rows_chunked = 0
    chunk_rows_created = 0

    original_total_rows = len(df)

    largest_original = 0
    largest_chunk = 0
    chunk_sizes = []

    for original_row_idx, row in df.iterrows():

        markers = [
            m.strip()
            for m in str(row["Marker_Genes"]).split(",")
            if m.strip()
        ]

        # ---------------------------------------
        # Shuffle markers ONCE before chunking
        # ---------------------------------------

        random.shuffle(markers)

        marker_count = len(markers)

        largest_original = max(
            largest_original,
            marker_count
        )

        # ---------------------------------------
        # Normal row
        # ---------------------------------------

        if marker_count <= THRESHOLD:

            new_rows.append({

                "Original_Row": original_row_idx,

                "Cell_ID": row["Cell_ID"],

                "Cell_Name": row["Cell_Name"],

                "Cell_Name_Extended": row["Cell_Name_Extended"],

                "Tissue": row["Tissue"],

                "Chunk_ID": 0,

                "Marker_Genes": ",".join(markers),

                "Marker_Count": marker_count

            })

            largest_chunk = max(
                largest_chunk,
                marker_count
            )

            chunk_sizes.append(marker_count)

            continue

        # ---------------------------------------
        # Chunk large rows
        # ---------------------------------------

        rows_chunked += 1

        chunks = chunk_markers(markers)

        for chunk_id, chunk in enumerate(chunks):

            size = len(chunk)

            largest_chunk = max(
                largest_chunk,
                size
            )

            chunk_sizes.append(size)

            new_rows.append({

                "Original_Row": original_row_idx,

                "Cell_ID": row["Cell_ID"],

                "Cell_Name": row["Cell_Name"],

                "Cell_Name_Extended": row["Cell_Name_Extended"],

                "Tissue": row["Tissue"],

                "Chunk_ID": chunk_id,

                "Marker_Genes": ",".join(chunk),

                "Marker_Count": size

            })

        chunk_rows_created += len(chunks)

    result = pd.DataFrame(new_rows)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========== CHUNKING SUMMARY ==========")
    print(f"Random seed                  : {RANDOM_SEED}")
    print(f"Original rows                : {original_total_rows}")
    print(f"Rows exceeding {THRESHOLD} markers : {rows_chunked}")
    print(f"Chunk rows created           : {chunk_rows_created}")
    print(f"Final dataset rows           : {len(result)}")
    print(f"Largest original marker set  : {largest_original}")
    print(f"Largest chunk size           : {largest_chunk}")
    print(f"Average chunk size           : {sum(chunk_sizes)/len(chunk_sizes):.2f}")
    print(f"Saved to                     : {OUTPUT_PATH}")
    print("======================================\n")


if __name__ == "__main__":
    main()