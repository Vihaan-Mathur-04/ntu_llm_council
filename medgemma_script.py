"""
Standalone MedGemma 4B Cell Annotation
NTU LLM Council Project

PURPOSE
-------
Standalone MedGemma experiment.

MedGemma receives only the marker genes for each cell.
It does not receive:
- Other model predictions
- Council outputs
- RAG evidence
- CellMarker retrieval

The resulting CSV contains the original dataset plus
MedGemma's prediction, confidence, reasoning, and raw output.
"""

import pandas as pd
import time
import gc
import torch

from backend.model_loader import run_unified_model


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = "table_format.csv"
OUTPUT_FILE = "medgemma_new_predictions.csv"

MARKER_COLUMN = "Marker_Genes"


# =========================================================
# PROMPT
# =========================================================

def format_prompt(marker_text):

    return f"""
You are an expert computational biologist specializing in
single-cell RNA sequencing (scRNA-seq) cell type annotation.

You will be given the marker genes associated with one
biological cell.

Your task is to identify the single most likely cell type.

==================================================

MARKER GENES

{marker_text}

==================================================

ANALYSIS

Analyze the complete marker profile.

The marker genes are presented in arbitrary order and their
order has no biological meaning.

Consider:

1. Lineage-defining markers.
2. Cell-type-specific markers.
3. Supporting markers.
4. Conflicting or nonspecific markers.
5. The overall biological coherence of the marker profile.

Do not rely on a single marker unless it is highly specific.

Use your own biological knowledge to determine the most
likely cell type.

==================================================

OUTPUT

Return exactly:

LABEL: <cell type>
CONFIDENCE: <number between 0 and 1>
REASONING: <brief biological explanation>

Rules:

- Output only these three fields.
- Give exactly one cell type.
- Do not provide alternative predictions.
- Do not mention other models.
- Do not mention a council.
- Do not mention RAG or retrieval.
- Do not output markdown.
""".strip()


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(df):

    pipeline_start = time.perf_counter()

    predictions = []
    confidences = []
    reasoning = []
    raw_outputs = []

    # =====================================================
    # PROCESS EACH CELL
    # =====================================================

    for i in range(len(df)):

        cell_start = time.perf_counter()

        marker_text = str(
            df.iloc[i][MARKER_COLUMN]
        )

        prompt = format_prompt(marker_text)

        print("\n========================================")
        print(f"CELL {i + 1}/{len(df)}")
        print("========================================")

        output = run_unified_model(
            model_name="medgemma",
            prompt=prompt,
            max_new_tokens=128
        )

        print("\nRAW MEDGEMMA OUTPUT:")
        print(repr(output))

        # -------------------------------------------------
        # Parse response
        # -------------------------------------------------

        label = "ERROR"
        confidence = None
        cell_reasoning = ""

        if isinstance(output, str):

            if "LABEL:" in output:

                label = (
                    output
                    .split("LABEL:", 1)[1]
                    .split("\n", 1)[0]
                    .strip()
                )

            if "CONFIDENCE:" in output:

                confidence_text = (
                    output
                    .split("CONFIDENCE:", 1)[1]
                    .split("\n", 1)[0]
                    .strip()
                )

                try:
                    confidence = float(
                        confidence_text
                    )
                except ValueError:
                    confidence = None

            if "REASONING:" in output:

                cell_reasoning = (
                    output
                    .split("REASONING:", 1)[1]
                    .strip()
                )

        predictions.append(label)
        confidences.append(confidence)
        reasoning.append(cell_reasoning)
        raw_outputs.append(
            output if isinstance(output, str)
            else "ERROR"
        )

        cell_time = (
            time.perf_counter()
            - cell_start
        )

        print(
            f"[MEDGEMMA][{i}] → {label} "
            f"({cell_time:.2f}s)"
        )

    # =====================================================
    # TIMING
    # =====================================================

    total_runtime = (
        time.perf_counter()
        - pipeline_start
    )

    # =====================================================
    # OUTPUT
    # =====================================================

    output_df = df.copy()

    output_df["MedGemma"] = predictions
    output_df["MedGemma_Confidence"] = confidences
    output_df["MedGemma_Reasoning"] = reasoning
    output_df["MedGemma_Raw"] = raw_outputs

    output_df["Total_Runtime_Seconds"] = total_runtime

    # =====================================================
    # CLEANUP
    # =====================================================

    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # =====================================================
    # SAVE
    # =====================================================

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n========================================")
    print("MEDGEMMA EXPERIMENT COMPLETE")
    print("========================================")

    print(
        f"Cells processed : {len(df)}"
    )

    print(
        f"Total runtime   : "
        f"{total_runtime:.2f} seconds"
    )

    print(
        f"Total runtime   : "
        f"{total_runtime / 60:.2f} minutes"
    )

    print(
        f"Average/cell    : "
        f"{total_runtime / len(df):.2f} seconds"
        if len(df) > 0
        else "Average/cell    : N/A"
    )

    print(
        f"Output file     : "
        f"{OUTPUT_FILE}"
    )

    print("========================================")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("[INFO] Loading dataset...")

    df = pd.read_csv(INPUT_FILE).copy()

    print(
        f"[INFO] Loaded {len(df)} rows."
    )

    run_pipeline(df)