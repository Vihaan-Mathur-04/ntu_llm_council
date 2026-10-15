"""
Evaluate RAG Retrieval Accuracy

Outputs
-------
- Top-1 accuracy
- Top-2 accuracy
- Top-3 accuracy
- Top-4 accuracy
- Ground-truth rank
- Dice similarity scores

Saves:
retrieval_accuracy.csv
"""

import pandas as pd

from rag_retriever import retrieve_candidates


# =====================================================
# LOAD DATA
# =====================================================

CSV_PATH = "/Users/vihaan_mathur/ntu_project/llm_council/cellstar_preprocessed_chunked.csv"

df = pd.read_csv(CSV_PATH)


# =====================================================
# NORMALIZATION
# =====================================================

def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


# =====================================================
# METRICS
# =====================================================

results = []
failure_results = []

top1_correct = 0
top2_correct = 0
top3_correct = 0
top4_correct = 0

rank_counts = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    "not_found": 0
}


# =====================================================
# MAIN LOOP
# =====================================================

for idx, row in df.iterrows():

    # ---------------------------------
    # Marker genes
    # ---------------------------------

    marker_string = row["Marker_Genes"]

    if pd.isna(marker_string):
        continue

    query_genes = [
        gene.strip()
        for gene in str(marker_string).split(",")
        if gene.strip()
    ]

    # ---------------------------------
    # Ground truth
    # ---------------------------------

    ground_truth = str(row["Cell_Name"]).strip()
    gt_norm = normalize(ground_truth)

    # ---------------------------------
    # Retrieve
    # ---------------------------------

    retrievals = retrieve_candidates(
        query_markers=query_genes,
        top_k=4
    )

    retrieved_names = []
    retrieved_scores = []
    best_dice = 0.0

    gt_rank = None
    gt_dice = None

    for rank, item in enumerate(retrievals, start=1):

        name = item["cell_name"]
        score = item["dice_score"]

        retrieved_names.append(name)
        retrieved_scores.append(score)

        if (
            gt_rank is None
            and normalize(name) == gt_norm
        ):
            gt_rank = rank
            gt_dice = score
    
    if len(retrieved_scores) > 0:
        best_dice = retrieved_scores[0]
    # ---------------------------------
    # Pad to Top-4
    # ---------------------------------

    while len(retrieved_names) < 4:
        retrieved_names.append("")
        retrieved_scores.append("")

    retrieved_norm = [
        normalize(x)
        for x in retrieved_names
    ]

    # ---------------------------------
    # Accuracy
    # ---------------------------------

    top1 = gt_norm == retrieved_norm[0]
    top2 = gt_norm in retrieved_norm[:2]
    top3 = gt_norm in retrieved_norm[:3]
    top4 = gt_norm in retrieved_norm[:4]

    if not top4:

        failure_results.append({

            "query_index": idx,

            "ground_truth": ground_truth,

            "query_marker_count":
                len(query_genes),

            "query_markers":
                ", ".join(query_genes),

            "top1":
                retrieved_names[0],

            "dice1":
                retrieved_scores[0],

            "top2":
                retrieved_names[1],

            "dice2":
                retrieved_scores[1],

            "top3":
                retrieved_names[2],

            "dice3":
                retrieved_scores[2],

            "top4":
                retrieved_names[3],

            "dice4":
                retrieved_scores[3],

            "best_dice": best_dice,
            "ground_truth_dice": gt_dice
        })

    if top1:
        top1_correct += 1

    if top2:
        top2_correct += 1

    if top3:
        top3_correct += 1

    if top4:
        top4_correct += 1

    # ---------------------------------
    # Rank statistics
    # ---------------------------------

    if gt_rank is None:
        rank_counts["not_found"] += 1
    else:
        rank_counts[gt_rank] += 1

    # ---------------------------------
    # Save row
    # ---------------------------------

    results.append({

        "query_index": idx,

        "ground_truth": ground_truth,

        "top1": retrieved_names[0],
        "dice1": retrieved_scores[0],

        "top2": retrieved_names[1],
        "dice2": retrieved_scores[1],

        "top3": retrieved_names[2],
        "dice3": retrieved_scores[2],

        "top4": retrieved_names[3],
        "dice4": retrieved_scores[3],

        "ground_truth_rank": gt_rank,

        "top1_correct": top1,
        "top2_correct": top2,
        "top3_correct": top3,
        "top4_correct": top4,

        "best_dice": best_dice,
        "ground_truth_dice": gt_dice

    })


# =====================================================
# SAVE CSV
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "retrieval_accuracy.csv",
    index=False
)

failure_df = pd.DataFrame(failure_results)

failure_df.to_csv(
    "retrieval_failures.csv",
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

n = len(results_df)

print("\n==============================")
print("RAG RETRIEVAL RESULTS")
print("==============================")
print(f"Samples        : {n}")
print(f"Top-1 Accuracy : {top1_correct / n:.3f}")
print(f"Top-2 Accuracy : {top2_correct / n:.3f}")
print(f"Top-3 Accuracy : {top3_correct / n:.3f}")
print(f"Top-4 Accuracy : {top4_correct / n:.3f}")
print("------------------------------")
print("Ground Truth Rank Distribution")
print(f"Rank 1         : {rank_counts[1]}")
print(f"Rank 2         : {rank_counts[2]}")
print(f"Rank 3         : {rank_counts[3]}")
print(f"Rank 4         : {rank_counts[4]}")
print(f"Not Retrieved  : {rank_counts['not_found']}")
dice_series = results_df["best_dice"]

print("------------------------------")
print("Best Dice Statistics")
print(f"Maximum      : {dice_series.max():.3f}")
print(f"Mean         : {dice_series.mean():.3f}")
print(f"Median       : {dice_series.median():.3f}")
print(f"Minimum      : {dice_series.min():.3f}")

print("------------------------------")
print("Best Dice Percentiles")
print(f"25%          : {dice_series.quantile(0.25):.3f}")
print(f"50%          : {dice_series.quantile(0.50):.3f}")
print(f"75%          : {dice_series.quantile(0.75):.3f}")
print(f"90%          : {dice_series.quantile(0.90):.3f}")
print(f"95%          : {dice_series.quantile(0.95):.3f}")
thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

print("------------------------------")
print("Threshold Analysis")

for t in thresholds:
    kept = (results_df["best_dice"] >= t).sum()
    print(f"{t:.2f} -> {kept}/{n} retrieved ({100*kept/n:.1f}%)")

print("==============================")