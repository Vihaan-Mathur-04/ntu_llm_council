"""
Cell annotation pipeline (NTU LLM Council - MEMORY SAFE VERSION)
"""

import asyncio
import time
import pandas as pd

from backend.original_council import run_full_council
from backend.original_council import safe_json_parse


# =========================================================
# PROMPT
# =========================================================

def format_prompt(marker_text: str):

    return f"""
You are an expert computational biologist specializing in single-cell RNA sequencing (scRNA-seq) cell type annotation.

==================================================

MARKER GENES

{marker_text}

==================================================

Your task is to identify the single most likely biological cell type represented by these marker genes.

The marker genes are presented in arbitrary order.

• The order has NO biological meaning.
• Evaluate the COMPLETE marker profile before reaching a conclusion.
• Consider all markers collectively.
• Do not rely on only one or two genes.

==================================================

BIOLOGICAL ANALYSIS

1. Interpret the marker genes independently.
2. Identify the most informative lineage-defining markers.
3. Infer the most likely biological cell type.
4. Explain your reasoning.
5. Estimate your confidence.

==================================================

GUIDELINES

• Base your decision ONLY on the supplied marker genes.
• Mention only the most informative biological markers. You can only mention 5 - 7 marker genes in your reasoning.
• Keep the reasoning concise (3–5 sentences).
• Confidence should reflect biological certainty.

==================================================

OUTPUT

Return EXACTLY ONE JSON object.

{{
  "prediction":"The most likely cell type",
  "confidence":0.00,
  "reasoning":"Concise biological explanation."
}}

IMPORTANT

• Return ONE JSON object only.
• No markdown.
• No code fences.
• No explanations before the JSON.
• No explanations after the JSON.
• Stop immediately after the closing brace.
""".strip()

# =========================================================
# PIPELINE
# =========================================================

async def run_pipeline(df):

    pipeline_start = time.time()

    marker_col = df.columns[5]

    # -----------------------------------------------------
    # Council predictions
    # -----------------------------------------------------

    phi_outputs = []
    llama_outputs = []
    txgemma_outputs = []

    # -----------------------------------------------------
    # Council reasoning
    # -----------------------------------------------------

    phi_reasoning = []
    llama_reasoning = []
    txgemma_reasoning = []

    # -----------------------------------------------------
    # Raw outputs
    # -----------------------------------------------------

    phi_raw = []
    llama_raw = []
    txgemma_raw = []

    # -----------------------------------------------------
    # Chairman outputs
    # -----------------------------------------------------

    final_labels = []
    final_confidences = []
    final_reasoning = []
    final_raw = []

    # -----------------------------------------------------
    # Agreement tracking
    # -----------------------------------------------------

    chairman_followed_rag = []
    chairman_followed_majority = []

    # =====================================================

    for i in range(len(df)):

        print("\n================================")
        print(f"CELL {i}")
        print("================================\n")

        marker_text = str(df.iloc[i][marker_col])

        stage1_results, chairman_result, chairman_raw, metadata = await run_full_council(
            user_query=format_prompt(marker_text)
        )

        print("\nSTAGE 1 RESULTS:")
        print(stage1_results)

        print("\nCHAIRMAN RESULT:")
        print(chairman_result)

        # -------------------------------------------------
        # Individual model outputs
        # -------------------------------------------------

        phi_result = next(
            (x for x in stage1_results if x["model"] == "phi"),
            {}
        )

        llama_result = next(
            (x for x in stage1_results if x["model"] == "llama"),
            {}
        )

        txgemma_result = next(
            (x for x in stage1_results if x["model"] == "txgemma"),
            {}
        )

        # Predictions

        phi_outputs.append(phi_result.get("prediction", "ERROR"))
        llama_outputs.append(llama_result.get("prediction", "ERROR"))
        txgemma_outputs.append(txgemma_result.get("prediction", "ERROR"))

        # Reasoning

        phi_reasoning.append(phi_result.get("reasoning", ""))
        llama_reasoning.append(llama_result.get("reasoning", ""))
        txgemma_reasoning.append(txgemma_result.get("reasoning", ""))

        # Raw outputs

        phi_raw.append(phi_result.get("raw_response", ""))
        llama_raw.append(llama_result.get("raw_response", ""))
        txgemma_raw.append(txgemma_result.get("raw_response", ""))

        # -------------------------------------------------
        # Chairman
        # -------------------------------------------------

        if isinstance(chairman_result, str):
            chairman_result = safe_json_parse(chairman_result)

        if chairman_result is None:
            chairman_result = {
                "prediction": "ERROR",
                "confidence": 0.0,
                "reasoning": "No chairman response"
            }

        final_prediction = chairman_result.get(
            "prediction",
            "ERROR"
        )

        final_labels.append(final_prediction)
        final_confidences.append(
            chairman_result.get("confidence", None)
        )
        final_reasoning.append(
            chairman_result.get("reasoning", "")
        )

        final_raw.append(chairman_raw)

        # -------------------------------------------------
        # Agreement
        # -------------------------------------------------


        chairman_followed_majority.append(
            metadata.get(
                "chairman_followed_majority",
                False
            )
        )

    # =====================================================
    # FINAL DATAFRAME
    # =====================================================

    output_df = df.copy()

    # -------------------------------
    # Model predictions
    # -------------------------------

    output_df["Phi"] = phi_outputs
    output_df["Llama"] = llama_outputs
    output_df["TxGemma"] = txgemma_outputs

    # -------------------------------
    # Model reasoning
    # -------------------------------

    output_df["Phi_Reasoning"] = phi_reasoning
    output_df["Llama_Reasoning"] = llama_reasoning
    output_df["TxGemma_Reasoning"] = txgemma_reasoning

    # -------------------------------
    # Raw outputs
    # -------------------------------

    output_df["Phi_Raw"] = phi_raw
    output_df["Llama_Raw"] = llama_raw
    output_df["TxGemma_Raw"] = txgemma_raw

    # -------------------------------
    # Final council output
    # -------------------------------

    output_df["Council_Final"] = final_labels
    output_df["Council_Confidence"] = final_confidences
    output_df["Council_Reasoning"] = final_reasoning
    output_df["Council_Raw"] = final_raw

    # -------------------------------
    # Chairman agreement
    # -------------------------------

    output_df["Chairman_Followed_Majority"] = chairman_followed_majority

    print("\n========================")
    print("DONE")
    print("========================")
    print(
        f"TOTAL TIME: {(time.time() - pipeline_start)/60:.2f} minutes"
    )
    total_runtime = time.time() - pipeline_start

    output_df["Total_Runtime_Seconds"] = total_runtime

    return output_df


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print("[INFO] This module is now pipeline-only (no direct execution).")