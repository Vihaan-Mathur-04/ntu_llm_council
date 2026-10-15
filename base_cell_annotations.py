"""
Cell annotation pipeline (NTU LLM Council - MEMORY SAFE VERSION)
"""

import asyncio
import time
import pandas as pd

from backend.base_council import run_full_council
from backend.base_council import safe_json_parse


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

Your task is to identify the single most likely cell type represented by these marker genes.

The marker genes are presented in arbitrary order.

• The order has NO biological meaning.
• Evaluate the COMPLETE marker profile before reaching a conclusion.
• Do not assume that one portion of the marker list is more important than another.

==================================================

RETRIEVED EVIDENCE

You will also receive candidate cell types retrieved from the CellMarker database.

The retrieved candidates are supporting evidence only, NOT ground truth.

• Candidates are ranked using an IDF-weighted Dice similarity score.
• Higher similarity indicates stronger biological similarity but does NOT guarantee correctness.
• The correct cell type may not appear among the retrieved candidates.
• Use the retrieval to support or challenge your own biological interpretation rather than as the answer itself.

==================================================

BIOLOGICAL ANALYSIS

1. Interpret the marker genes independently.
2. Identify the most informative lineage-defining markers.
3. Infer the most likely cell type before considering the retrieved candidates.
4. Compare your conclusion with the retrieved evidence.
5. Decide whether the retrieval supports or contradicts your biological interpretation.
6. Produce the final prediction.

==================================================

GUIDELINES

• Do NOT automatically choose the highest-ranked retrieved candidate.
• Do NOT assume the retrieved candidates contain the correct answer.
• If the retrieval disagrees with your biological interpretation, explain why.
• Base your conclusion on the complete marker profile rather than isolated genes.
• Mention only the 3–5 most biologically informative marker genes.
• Do not reproduce long lists of marker genes.
• Set confidence according to the biological evidence, not retrieval agreement.
• Keep the reasoning concise (approximately 3–5 sentences).

==================================================

OUTPUT

Return EXACTLY ONE JSON object.

The JSON object MUST contain all three fields shown below.

Fill in every field.

The "prediction" field must contain your predicted biological cell type.

The "reasoning" field must briefly justify your prediction.

Return this exact structure:

{{
  "prediction":"The most likely cell type",
  "confidence":0.00,
  "reasoning":"Concise biological explanation."
}}

IMPORTANT

• Return ONE JSON object only.
• Do NOT generate multiple JSON objects.
• Do NOT revise or repeat your answer.
• Do NOT output markdown.
• Do NOT output code fences.
• Do NOT output explanations before the JSON.
• Do NOT output explanations after the JSON.
• Immediately stop generating after the final closing brace.""".strip()


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
    # Retrieval information
    # -----------------------------------------------------

    rag_top1 = []
    rag_top2 = []
    rag_top3 = []
    rag_top4 = []

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
            marker_list=marker_text,
            original_prompt=format_prompt(marker_text)
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
        # Retrieval metadata
        # -------------------------------------------------

        rag_top1.append(
            metadata.get("rag_top1", "")
        )

        rag_top2.append(
            metadata.get("rag_top2", "")
        )

        rag_top3.append(
            metadata.get("rag_top3", "")
        )

        rag_top4.append(
            metadata.get("rag_top4", "")
        )

        # -------------------------------------------------
        # Agreement
        # -------------------------------------------------

        chairman_followed_rag.append(
            metadata.get(
                "chairman_followed_any_rag",
                False
            )
        )

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
    # RAG retrieval
    # -------------------------------

    output_df["RAG_Top1"] = rag_top1
    output_df["RAG_Top2"] = rag_top2
    output_df["RAG_Top3"] = rag_top3
    output_df["RAG_Top4"] = rag_top4

    # -------------------------------
    # Chairman agreement
    # -------------------------------

    output_df["Chairman_Followed_RAG"] = chairman_followed_rag
    output_df["Chairman_Followed_Majority"] = chairman_followed_majority

    print("\n========================")
    print("DONE")
    print("========================")
    print(
        f"TOTAL TIME: {(time.time() - pipeline_start)/60:.2f} minutes"
    )

    return output_df


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print("[INFO] This module is now pipeline-only (no direct execution).")