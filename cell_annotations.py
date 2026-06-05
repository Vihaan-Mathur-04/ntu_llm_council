"""
Cell annotation pipeline (NTU LLM Council - MEMORY SAFE VERSION)

FINAL ARCHITECTURE:
- Load 1 model at a time
- Run ALL cells
- Unload model
- Repeat

Then:
- Load chairman
- Produce final classification + reasoning
"""

import asyncio
import pandas as pd
import torch
import gc

from backend.model_loader import run_unified_model


# =========================================================
# LOAD DATA (TEST MODE = 2 ROWS ONLY)
# =========================================================

def load_data(path: str):
    df = pd.read_csv(path).head(2)

    print(f"[INFO] Loaded shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    return df


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

Return ONLY the label.
""".strip()


# =========================================================
# MODEL OVER DATASET (SEQUENTIAL EXECUTION)
# =========================================================

async def run_model_over_dataset(df, model_name, marker_col):

    predictions = []

    print("\n====================")
    print(f"LOADING MODEL: {model_name}")
    print("====================\n")

    for i in range(len(df)):

        marker_text = str(df.iloc[i][marker_col])
        prompt = format_prompt(marker_text)

        try:
            output = run_unified_model(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=64
            )
        except Exception as e:
            print(f"[{model_name}][{i}] ERROR → {e}")
            output = "ERROR"

        if not isinstance(output, str) or len(output.strip()) == 0:
            output = "ERROR_EMPTY"

        label = output.strip()

        predictions.append({
            "label": label,
            "confidence": None
        })

        print(f"[{model_name}][{i}] → {label}")

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
                conf = output.split("CONFIDENCE:")[1].split("\n")[0].split("\n")[0].strip()

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
    if not model_outputs or len(model_outputs) == 0:
        return ["ERROR"] * n

    return [
        x["label"] if isinstance(x, dict) and "label" in x else "ERROR"
        for x in model_outputs
    ]


# =========================================================
# PIPELINE
# =========================================================

async def run_pipeline(input_path: str):

    df = load_data(input_path)
    marker_col = df.columns[2]

    base_outputs = {
        "txgemma": [],
        "qwen": [],
        "biomistral": []
    }

    # =====================================================
    # BASE MODELS (STRICT SEQUENTIAL EXECUTION)
    # =====================================================

    for model in ["txgemma", "qwen", "biomistral"]:

        preds = await run_model_over_dataset(df, model, marker_col)

        base_outputs[model] = preds

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # =====================================================
    # CHAIRMAN
    # =====================================================

    labels, confs, reasoning = await run_chairman(df, base_outputs, marker_col)

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # =====================================================
    # SAVE OUTPUT
    # =====================================================

    output_df = df.copy()

    output_df["TxGemma"] = safe_extract(base_outputs["txgemma"], len(df))
    output_df["Qwen"] = safe_extract(base_outputs["qwen"], len(df))
    output_df["BioMistral"] = safe_extract(base_outputs["biomistral"], len(df))

    output_df["Council_Final"] = labels
    output_df["Council_Confidence"] = confs
    output_df["Council_Reasoning"] = reasoning

    output_df.to_csv("cellstar_predictions.csv", index=False)

    print("\n========================")
    print("DONE")
    print("========================")
    print("Saved to: cellstar_predictions.csv")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    input_path = "/Users/vihaan_mathur/ntu_project/llm_council/cellstar_preprocessed.csv"
    asyncio.run(run_pipeline(input_path))