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

from backend.model_loader import run_unified_model


# =========================================================
# PROMPT
# =========================================================

def format_prompt(marker_text: str):

    return f"""
You are a biomedical expert classifier in single-cell transcriptomics.

INPUT:
Marker genes representing ONE cell type.

CRITICAL RULES:
- Output ONLY ONE label
- NO explanations
- NO gene repetition
- NO multi-line output

Marker genes:
{marker_text}

Valid labels:
T cell
B cell
NK cell
Monocyte
Macrophage
Dendritic cell
Endothelial cell
Stem cell

Return ONLY ONE label.
""".strip()


# =========================================================
# MODEL OVER DATASET (SEQUENTIAL EXECUTION)
# =========================================================

async def run_model_over_dataset(df, model_name, marker_col):

    predictions = []

    print("\n====================")
    print(f"LOADING MODEL: {model_name}")
    print("====================\n")

    batch_start = time.time()

    for i in range(len(df)):

        row_start = time.time()

        marker_text = str(df.iloc[i][marker_col])
        prompt = format_prompt(marker_text)

        output = run_unified_model(
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=64
        )

        label = output.strip() if isinstance(output, str) else "ERROR"

        predictions.append({
            "label": label,
            "confidence": None
        })

        row_time = time.time() - row_start
        print(f"[{model_name}][{i}] → {label} ({row_time:.2f}s)")

    batch_time = time.time() - batch_start
    print(f"\n[MODEL DONE] {model_name} took {batch_time:.2f}s\n")

    return predictions


# =========================================================
# CHAIRMAN PASS
# =========================================================

async def run_chairman(df, base_outputs, marker_col):

    final_labels = []
    final_confidences = []
    final_reasoning = []

    print("\n====================")
    print("LOADING CHAIRMAN (MEDGEMMA)")
    print("====================\n")

    for i in range(len(df)):

        prompt = f"""
You are the chairman of a biomedical LLM council.

Marker genes:
{df.iloc[i][marker_col]}

Base model predictions:
- TxGemma: {base_outputs['txgemma'][i]['label']}
- Qwen: {base_outputs['qwen'][i]['label']}
- BioMistral: {base_outputs['biomistral'][i]['label']}

TASK:
1. Compare predictions
2. Resolve disagreement
3. Output final label
4. Provide confidence (0-1)
5. Provide short reasoning

FORMAT:
LABEL: ...
CONFIDENCE: ...
REASONING: ...
""".strip()

        output = run_unified_model(
            model_name="medgemma",
            prompt=prompt,
            max_new_tokens=128
        )

        label = "ERROR"
        conf = None
        reasoning = output

        if isinstance(output, str):
            if "LABEL:" in output:
                label = output.split("LABEL:")[1].split("\n")[0].strip()

            if "CONFIDENCE:" in output:
                conf = output.split("CONFIDENCE:")[1].split("\n")[0].strip()

            if "REASONING:" in output:
                reasoning = output.split("REASONING:")[1].strip()

        final_labels.append(label)
        final_confidences.append(conf)
        final_reasoning.append(reasoning)

        print(f"[CHAIR][{i}] → {label}")

    return final_labels, final_confidences, final_reasoning


# =========================================================
# SAFE EXTRACTOR
# =========================================================

def safe_extract(model_outputs, n):
    if not model_outputs:
        return ["ERROR"] * n

    return [
        x["label"] if isinstance(x, dict) and "label" in x else "ERROR"
        for x in model_outputs
    ]


# =========================================================
# PIPELINE (ONLY ACCEPTS DF)
# =========================================================

async def run_pipeline(df):

    pipeline_start = time.time()
    marker_col = df.columns[2]

    base_outputs = {
        "txgemma": [],
        "qwen": [],
        "biomistral": []
    }

    # -------------------------
    # BASE MODELS
    # -------------------------
    for model in ["txgemma", "qwen", "biomistral"]:

        preds = await run_model_over_dataset(df, model, marker_col)
        base_outputs[model] = preds

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # -------------------------
    # CHAIRMAN
    # -------------------------
    labels, confs, reasoning = await run_chairman(df, base_outputs, marker_col)

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    output_df = df.copy()

    output_df["TxGemma"] = safe_extract(base_outputs["txgemma"], len(df))
    output_df["Qwen"] = safe_extract(base_outputs["qwen"], len(df))
    output_df["BioMistral"] = safe_extract(base_outputs["biomistral"], len(df))

    output_df["Council_Final"] = labels
    output_df["Council_Confidence"] = confs
    output_df["Council_Reasoning"] = reasoning

    print("\n========================")
    print("DONE")
    print("========================")
    print(f"TOTAL TIME: {(time.time() - pipeline_start)/60:.2f} minutes")

    return output_df


# =========================================================
# ENTRY POINT (SAFE TEST ONLY)
# =========================================================

if __name__ == "__main__":

    # IMPORTANT:
    # DO NOT call run_pipeline with arguments anymore.
    # Use batch_runner.py for slicing.

    print("[INFO] This module is now pipeline-only (no direct execution).")