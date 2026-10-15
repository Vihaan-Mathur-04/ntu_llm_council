"""
RAG Base Setup
==============

Phase 1:
- Load CellMarker 2.0
- Filter to Human entries
- Explore dataset structure
- Build grouped cell-type corpus
- Save corpus for retrieval

Usage:
    python rag_base_setup.py
"""

from pathlib import Path
import math
import pandas as pd
import json

from sympy import Sum


# ============================================================
# CONFIG
# ============================================================

CELLMARKER_FILE = "/Users/vihaan_mathur/ntu_project/llm_council/CELLMARKER RAG SET.xlsx"
OUTPUT_CORPUS = "cellmarker_corpus.json"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING CELLMARKER")
print("=" * 80)

df = pd.read_excel(CELLMARKER_FILE)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f" - {col}")


# ============================================================
# SPECIES ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("SPECIES DISTRIBUTION")
print("=" * 80)

print(df["species"].value_counts(dropna=False))

human_df = df[df["species"].astype(str).str.lower() == "human"]

# ============================================================
# BUILD GLOBAL MARKER FREQUENCIES
# ============================================================

print("\n" + "=" * 80)
print("BUILDING GLOBAL MARKER STATISTICS")
print("=" * 80)

marker_to_cells = {}

for _, row in human_df.iterrows():

    marker = str(row["marker"]).strip().upper()

    cell = str(row["cell_name"]).strip()

    if marker == "" or marker == "NAN":
        continue

    marker_to_cells.setdefault(marker, set()).add(cell)

total_cells = human_df["cell_name"].nunique()

marker_idf = {}

for marker, cells in marker_to_cells.items():

    frequency = len(cells)

    idf = math.log((total_cells + 1) / (frequency + 1)) + 1

    marker_idf[marker] = {

        "frequency": frequency,

        "idf": round(idf, 4)

    }

print(f"Unique markers : {len(marker_idf):,}")

print(f"\nHuman rows retained: {len(human_df):,}")


# ============================================================
# CELL TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CELL TYPE ANALYSIS")
print("=" * 80)

num_cell_names = human_df["cell_name"].nunique()

print(f"Unique cell names: {num_cell_names:,}")

print("\nTop 20 most frequent cell names:\n")

print(
    human_df["cell_name"]
    .value_counts()
    .head(20)
)

print(f"Unique cell ontology IDs: {human_df['cellontology_id'].nunique():,}")

# ============================================================
# TISSUE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("TISSUE ANALYSIS")
print("=" * 80)

num_tissues = human_df["tissue_type"].nunique()

print(f"Unique tissues: {num_tissues:,}")

print("\nTop 20 tissues:\n")

print(
    human_df["tissue_type"]
    .value_counts()
    .head(20)
)


# ============================================================
# BUILD CORPUS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING CORPUS")
print("=" * 80)

corpus = []

grouped = human_df.groupby("cell_name")

for cell_name, group in grouped:

    marker_names = (
        group["marker"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    markers = {}

    for marker in sorted(marker_names):

        if marker in marker_idf:

            markers[marker] = marker_idf[marker]

    tissues = (
        group["tissue_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    pmids = (
        group["PMID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    journals = (
        group["journal"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    entry = {
        "cell_name": str(cell_name),

        "markers": markers,

        "marker_count": len(marker_names),

        "tissues": tissues,

        "pmids": pmids,

        "journals": journals,

        "evidence_count": len(group)
    }

    corpus.append(entry)


print(f"Corpus entries created: {len(corpus):,}")


# ============================================================
# SAVE CORPUS
# ============================================================

print("\n" + "=" * 80)
print("SAVING CORPUS")
print("=" * 80)

with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=2)

print(f"Saved corpus to: {OUTPUT_CORPUS}")


# ============================================================
# EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE CORPUS ENTRIES")
print("=" * 80)

for i, entry in enumerate(corpus[:5], start=1):

    print(f"\nEntry {i}")
    print("-" * 40)

    print(f"Cell Name: {entry['cell_name']}")
    print(f"Markers: {entry['marker_count']}")
    print(f"Evidence Count: {entry['evidence_count']}")

    preview = list(entry["markers"].keys())[:10]

    print(
        "Marker Preview:",
        ", ".join(preview)
    )


print("\nDone.")

Sum = sum(
    len(entry["markers"]) == 0
    for entry in corpus
)
print(Sum)
for entry in corpus:
    if entry["marker_count"] == 0:
        print(entry["cell_name"])
sorted(
    corpus,
    key=lambda x: x["marker_count"],
    reverse=True
)[:20]
import numpy as np
Mean = np.mean(
    [x["marker_count"] for x in corpus]
)
print(Mean)