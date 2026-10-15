import pandas as pd

# =====================================================
# CONFIG
# =====================================================

# Council output
PREDICTION_FILE = "medgemma_new_predictions.csv"

# Chunked CellStar reference containing ALL categories
REFERENCE_FILE = "cellstar_reference_preprocessed.csv"

# Which CellStar column to evaluate against
# Examples:
# "Cell_Name"
# "Cell_Name_Extended"
# "Tissue"

LABEL_COLUMN = "Tissue"

OUTPUT_FILE = f"ragless_accuracy_{LABEL_COLUMN}.csv"

# =====================================================
# LOAD FILES
# =====================================================

pred_df = pd.read_csv(PREDICTION_FILE)
ref_df = pd.read_csv(REFERENCE_FILE)

# =====================================================
# SAFETY CHECK
# =====================================================

if len(pred_df) != len(ref_df):
    raise ValueError(
        f"Row count mismatch!\n"
        f"Predictions: {len(pred_df)} rows\n"
        f"Reference : {len(ref_df)} rows"
    )

# =====================================================
# NORMALIZATION
# =====================================================

def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

ground_truth = normalize(ref_df[LABEL_COLUMN])

# =====================================================
# MODEL COLUMNS
# =====================================================

MODEL_COLUMNS = [
    "MedGemma"
]

MODEL_COLUMNS = [
    c for c in MODEL_COLUMNS
    if c in pred_df.columns
]

# =====================================================
# CALCULATE ACCURACIES
# =====================================================

results = []

for model in MODEL_COLUMNS:

    predictions = normalize(pred_df[model])

    correct = predictions == ground_truth

    results.append({

        "Model": model,

        "Reference Column": LABEL_COLUMN,

        "Correct Predictions": int(correct.sum()),

        "Total Cells": len(pred_df),

        "Accuracy": correct.mean()

    })

results = pd.DataFrame(results)

# =====================================================
# SAVE
# =====================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)

# =====================================================
# DISPLAY
# =====================================================

print("\n===================================")
print("HIERARCHICAL ACCURACY RESULTS")
print("===================================\n")

print(f"Prediction file : {PREDICTION_FILE}")
print(f"Reference file  : {REFERENCE_FILE}")
print(f"Ground truth    : {LABEL_COLUMN}\n")

print(results)

print(f"\nSaved to {OUTPUT_FILE}")