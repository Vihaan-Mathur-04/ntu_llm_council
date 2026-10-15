"""
Cell annotation pipeline (NTU LLM Council - MEMORY SAFE VERSION)
"""

import asyncio
import time
import pandas as pd

from backend.chairman_council import run_full_council
from backend.chairman_council import safe_json_parse


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

    initial_predictions = []
    initial_confidences = []
    initial_rationales = []

    initial_raw = []
    final_raw = []

    chairman_changed_prediction = []

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

        (
            council_reviews,
            chairman_initial,
            chairman_initial_raw,
            chairman_final,
            chairman_final_raw,
            metadata
        ) = await run_full_council(
        marker_list=marker_text,
        original_prompt=""
        )

        print("\nCOUNCIL REVIEWS:")
        print(council_reviews)

        print("\nINITIAL CHAIRMAN:")
        print(chairman_initial)
        initial_raw.append(chairman_initial_raw)
        initial_predictions.append(
            chairman_initial.get("prediction", "ERROR")
        )

        initial_confidences.append(
            chairman_initial.get("confidence", None)
        )

        initial_rationales.append(
            chairman_initial.get("rationale", "")
        )

        print("\nFINAL CHAIRMAN:")
        print(chairman_final)
        final_raw.append(chairman_final_raw)

        # -------------------------------------------------
        # Individual model outputs
        # -------------------------------------------------

        phi_result = next(
            (x for x in council_reviews if x["model"] == "phi"),
            {}
        )

        llama_result = next(
            (x for x in council_reviews if x["model"] == "llama"),
            {}
        )

        txgemma_result = next(
            (x for x in council_reviews if x["model"] == "txgemma"),
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

        if isinstance(chairman_final, str):
            chairman_final = safe_json_parse(chairman_final)
        
        chairman_changed_prediction.append(
            chairman_final.get("changed_prediction", False)
        )

        final_prediction = chairman_final.get(
            "prediction",
            "ERROR"
        )

        final_labels.append(final_prediction)

        final_confidences.append(
            chairman_final.get(
                "confidence",
                None
            )
        )

        final_reasoning.append(
            chairman_final.get(
                "reasoning",
                ""
            )
        )

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
                "chairman_followed_rag",
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
    # Chairman Initial
    # -------------------------------

    output_df["Chairman_Initial"] = initial_predictions
    output_df["Chairman_Initial_Confidence"] = initial_confidences
    output_df["Chairman_Initial_Rationale"] = initial_rationales
    output_df["Chairman_Initial_Raw"] = initial_raw

    # -------------------------------
    # Phi
    # -------------------------------

    output_df["Phi"] = phi_outputs
    output_df["Phi_Reasoning"] = phi_reasoning
    output_df["Phi_Raw"] = phi_raw

    # -------------------------------
    # TxGemma
    # -------------------------------

    output_df["TxGemma"] = txgemma_outputs
    output_df["TxGemma_Reasoning"] = txgemma_reasoning
    output_df["TxGemma_Raw"] = txgemma_raw

    # -------------------------------
    # Llama
    # -------------------------------

    output_df["Llama"] = llama_outputs
    output_df["Llama_Reasoning"] = llama_reasoning
    output_df["Llama_Raw"] = llama_raw

    # -------------------------------
    # Chairman Final
    # -------------------------------

    output_df["Chairman_Final"] = final_labels
    output_df["Chairman_Final_Confidence"] = final_confidences
    output_df["Chairman_Final_Reasoning"] = final_reasoning
    output_df["Chairman_Final_Raw"] = final_raw

    output_df["Chairman_Changed_Prediction"] = chairman_changed_prediction

    # -------------------------------
    # Retrieval
    # -------------------------------

    output_df["RAG_Top1"] = rag_top1
    output_df["RAG_Top2"] = rag_top2
    output_df["RAG_Top3"] = rag_top3
    output_df["RAG_Top4"] = rag_top4

    # -------------------------------
    # Agreement
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