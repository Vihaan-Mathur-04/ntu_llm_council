"""
Cell annotation pipeline (NTU LLM Council - MEMORY SAFE VERSION)

FINAL ARCHITECTURE:
- Input: dataframe ONLY (NO file handling here)
- Load 1 model at a time
- Run ALL cells
- Unload model
- Repeat

Then:
- Load chairman
- Produce final classification + reasoning
"""

import asyncio
import time
import pandas as pd
import torch
import gc

from backend.base_centric_rag import build_base_centric_prompt
from backend.base_council import run_full_council
from backend.base_council import run_full_council

# =========================================================
# PROMPT
# =========================================================

def format_prompt(marker_text: str):

    return f"""
You are a computational biologist specializing in single-cell RNA sequencing.

Marker genes from ONE biological cell:

{marker_text}

Determine the most likely biological cell type.

Provide:
1. Cell type label
2. Confidence score (0-1)
3. Short biological justification
""".strip()


# =========================================================
# PIPELINE (ONLY ACCEPTS DF)
# =========================================================

async def run_pipeline(df):

    pipeline_start = time.time()

    marker_col = df.columns[2]

    final_labels = []
    final_confidences = []
    final_reasoning = []

    qwen_outputs = []
    meditron_outputs = []

    for i in range(len(df)):

        print("\n================================")
        print(f"CELL {i}")
        print("================================\n")

        marker_text = str(df.iloc[i][marker_col])

        stage1_results, chairman_result, metadata = await run_full_council(
            marker_list=marker_text,
            original_prompt=format_prompt(marker_text)
        )

        # -----------------------------------
        # Store council member outputs
        # -----------------------------------

        qwen_result = next(
            (x for x in stage1_results if x["model"] == "qwen"),
            {}
        )

        meditron_result = next(
            (x for x in stage1_results if x["model"] == "meditron"),
            {}
        )

        qwen_outputs.append(
            qwen_result.get("prediction", "ERROR")
        )

        meditron_outputs.append(
            meditron_result.get("prediction", "ERROR")
        )

        # -----------------------------------
        # Store chairman outputs
        # -----------------------------------

        final_labels.append(
            chairman_result.get("prediction", "ERROR")
        )

        final_confidences.append(
            chairman_result.get("confidence", None)
        )

        final_reasoning.append(
            chairman_result.get("reasoning", "")
        )

    output_df = df.copy()

    output_df["Qwen"] = qwen_outputs
    output_df["Meditron"] = meditron_outputs

    output_df["Council_Final"] = final_labels
    output_df["Council_Confidence"] = final_confidences
    output_df["Council_Reasoning"] = final_reasoning

    print("\n========================")
    print("DONE")
    print("========================")
    print(
        f"TOTAL TIME: {(time.time() - pipeline_start)/60:.2f} minutes"
    )

    return output_df


# =========================================================
# ENTRY POINT (SAFE TEST ONLY)
# =========================================================

if __name__ == "__main__":

    # IMPORTANT:
    # DO NOT call run_pipeline with arguments anymore.
    # Use batch_runner.py for slicing.

    print("[INFO] This module is now pipeline-only (no direct execution).")