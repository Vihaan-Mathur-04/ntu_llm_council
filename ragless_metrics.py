import pandas as pd

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "ragless_new_cellstar_predictions.csv"
OUTPUT_FILE = "ragless_metrics.csv"

# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(INPUT_FILE)

# =====================================================
# BASIC
# =====================================================

total_cells = len(df)

correct = (
    df["Council_Final"].str.strip().str.lower()
    ==
    df["Cell_Name"].str.strip().str.lower()
)

overall_accuracy = correct.mean()

# =====================================================
# INDIVIDUAL MODEL ACCURACY
# =====================================================

phi_accuracy = (
    df["Phi"].str.strip().str.lower()
    ==
    df["Cell_Name"].str.strip().str.lower()
).mean()

llama_accuracy = (
    df["Llama"].str.strip().str.lower()
    ==
    df["Cell_Name"].str.strip().str.lower()
).mean()

txgemma_accuracy = (
    df["TxGemma"].str.strip().str.lower()
    ==
    df["Cell_Name"].str.strip().str.lower()
).mean()

# =====================================================
# AGREEMENT
# =====================================================

phi_llama = (
    df["Phi"].str.strip().str.lower()
    ==
    df["Llama"].str.strip().str.lower()
)

phi_tx = (
    df["Phi"].str.strip().str.lower()
    ==
    df["TxGemma"].str.strip().str.lower()
)

llama_tx = (
    df["Llama"].str.strip().str.lower()
    ==
    df["TxGemma"].str.strip().str.lower()
)

all_three = (
    phi_llama &
    phi_tx &
    llama_tx
)

# =====================================================
# MAJORITY
# =====================================================

majority_exists = (
    df["Chairman_Followed_Majority"]
    !=
    "No Majority"
)

chairman_followed = (
    df["Chairman_Followed_Majority"]
    == True
)

chairman_overruled = (
    df["Chairman_Followed_Majority"]
    == False
)

# =====================================================
# CONDITIONAL ACCURACY
# =====================================================

follow_accuracy = None

if chairman_followed.sum() > 0:

    follow_accuracy = (
        correct[chairman_followed]
        .mean()
    )

overrule_accuracy = None

if chairman_overruled.sum() > 0:

    overrule_accuracy = (
        correct[chairman_overruled]
        .mean()
    )

# =====================================================
# TIMING
# =====================================================

total_runtime = None
avg_time = None

if "Total_Runtime_Seconds" in df.columns:

    total_runtime = df["Total_Runtime_Seconds"].iloc[0]
    avg_time = total_runtime / total_cells

# =====================================================
# BUILD TABLE
# =====================================================

metrics = pd.DataFrame({

    "Metric":[

        "Total Cells",
        "Correct Predictions",
        "Overall Accuracy",

        "Phi Accuracy",
        "Llama Accuracy",
        "TxGemma Accuracy",

        "Phi-Llama Agreement",
        "Phi-TxGemma Agreement",
        "Llama-TxGemma Agreement",
        "All Three Agree",

        "Majority Exists",
        "Chairman Followed Majority",
        "Chairman Overruled Majority",

        "Accuracy (Follow Majority)",
        "Accuracy (Overrule Majority)",

        "Total Runtime (s)",
        "Average Time per Cell (s)"
    ],

    "Value":[

        total_cells,
        correct.sum(),
        overall_accuracy,

        phi_accuracy,
        llama_accuracy,
        txgemma_accuracy,

        phi_llama.mean(),
        phi_tx.mean(),
        llama_tx.mean(),
        all_three.mean(),

        majority_exists.sum(),
        chairman_followed.sum(),
        chairman_overruled.sum(),

        follow_accuracy,
        overrule_accuracy,

        total_runtime,
        avg_time
    ]

})

metrics.to_csv(
    OUTPUT_FILE,
    index=False
)

print(metrics)

print(f"\nSaved metrics to {OUTPUT_FILE}")