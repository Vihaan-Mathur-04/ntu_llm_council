"""
Base-Centric RAG Council
NTU Cell Annotation Project

PIPELINE

Stage 1:
    Shared RAG Retrieval

Stage 2:
    Independent Council Responses

Stage 3:
    Chairman Synthesis

Stage 4:
    Structured Return
"""

from typing import Dict
import asyncio
import json
import time
import re

from .config import (
    COUNCIL_MEMBERS,
    CHAIRMAN_MODEL
)

from .model_router import ModelRouter
from .rag import build_rag_prompt


# =====================================================
# JSON PARSER
# =====================================================

def safe_json_parse(text: str) -> Dict:
    """
    Robust JSON recovery.

    Handles:
    - empty responses
    - markdown fences
    - missing opening brace
    - missing closing brace
    - numbered outputs (e.g. '3. "prediction"...')
    - explanations before JSON
    - explanations after JSON
    - duplicated JSON objects
    - malformed JSON with intact keys
    """

    import json
    import re

    # --------------------------------------------------
    # Empty response
    # --------------------------------------------------

    if text is None or not str(text).strip():

        return {
            "prediction": "ERROR",
            "confidence": 0.0,
            "reasoning": "Empty response"
        }

    text = str(text or "").strip()

    # --------------------------------------------------
    # Remove markdown
    # --------------------------------------------------

    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = text.strip()

    # --------------------------------------------------
    # Remove leading numbering
    #
    # Example:
    # 3.
    # "prediction": ...
    # --------------------------------------------------

    text = re.sub(
        r"^\s*\d+\.\s*",
        "",
        text
    )
    # --------------------------------------------------
    # Repair missing opening brace
    # Example:
    #
    # 3.
    # "prediction": ...
    # --------------------------------------------------

    if (
        not text.startswith("{")
        and '"prediction"' in text
    ):
        text = "{\n" + text

    # --------------------------------------------------
    # Repair missing closing brace
    # --------------------------------------------------

    if (
        text.startswith("{")
        and not text.rstrip().endswith("}")
    ):
        text = text.rstrip() + "\n}"

    # --------------------------------------------------
    # Direct parse
    # --------------------------------------------------

    try:
        return json.loads(text)

    except Exception:
        pass

    # --------------------------------------------------
    # Locate prediction field
    # --------------------------------------------------

    prediction_pos = text.find('"prediction"')

    if prediction_pos != -1:

        repaired = text[prediction_pos:]

        # prepend brace if missing
        if not repaired.startswith("{"):
            repaired = "{" + repaired

        # balance braces
        if repaired.count("{") > repaired.count("}"):

            repaired += "}" * (
                repaired.count("{")
                - repaired.count("}")
            )

        try:

            parsed = json.loads(repaired)

            print("[PARSER] Recovered malformed JSON.")

            return parsed

        except Exception:
            pass

    # --------------------------------------------------
    # Extract FIRST complete JSON object
    # --------------------------------------------------

    start = text.find("{")

    if start != -1:

        depth = 0

        for i in range(start, len(text)):

            if text[i] == "{":
                depth += 1

            elif text[i] == "}":

                depth -= 1

                if depth == 0:

                    candidate = text[start:i+1]

                    try:

                        parsed = json.loads(candidate)

                        print("[PARSER] Extracted first JSON block.")

                        return parsed

                    except Exception:

                        break
    # --------------------------------------------------
    # TxGemma fallback parser
    #
    # Handles outputs like:
    #
    # 3. Ductal cell
    #    Weighted Dice Similarity : ...
    # --------------------------------------------------

    txgemma_match = re.search(
        r"""
        ^
        (?P<prediction>.+?)
        \n\s*Weighted\s+Dice\s+Similarity
        """,
        text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE
    )

    if txgemma_match:

        prediction = txgemma_match.group("prediction").strip()

        reasoning = ""

        reasoning_match = re.search(
            r"Shared\s+Markers\s*:.*?\n(?P<reasoning>.+)$",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if reasoning_match:

            reasoning = " ".join(
                reasoning_match.group("reasoning").split()
            )

        print("[PARSER] Parsed TxGemma numbered output.")

        return {
            "prediction": prediction,
            "confidence": 0.0,
            "reasoning": reasoning
        }
    # --------------------------------------------------
    # Last resort:
    #
    # Extract fields individually.
    #
    # This handles outputs like
    #
    # "prediction":"Macrophage",
    # "confidence":0.95,
    # ...
    # --------------------------------------------------

    try:

        prediction = re.search(
            r'"prediction"\s*:\s*"([^"]+)"',
            text
        )

        confidence = re.search(
            r'"confidence"\s*:\s*([0-9.]+)',
            text
        )

        agree = re.search(
            r'"agree_with_chairman"\s*:\s*(true|false)',
            text,
            flags=re.IGNORECASE
        )

        support = re.search(
            r'"supporting_evidence"\s*:\s*"([^"]*)"',
            text,
            flags=re.DOTALL
        )

        contradict = re.search(
            r'"contradicting_evidence"\s*:\s*"([^"]*)"',
            text,
            flags=re.DOTALL
        )

        reasoning = re.search(
            r'"reasoning"\s*:\s*"([^"]*)"',
            text,
            flags=re.DOTALL
        )

        if prediction:

            print("[PARSER] Reconstructed JSON from individual fields.")

            return {

                "prediction":
                    prediction.group(1),

                "confidence":
                    float(confidence.group(1))
                    if confidence else 0.0,

                "agree_with_chairman":
                    (
                        agree.group(1).lower() == "true"
                        if agree else False
                    ),

                "supporting_evidence":
                    support.group(1)
                    if support else "",

                "contradicting_evidence":
                    contradict.group(1)
                    if contradict else "",

                "reasoning":
                    reasoning.group(1)
                    if reasoning else ""
            }

    except Exception:
        pass
    # --------------------------------------------------
    # Final fallback
    # --------------------------------------------------

    return {
        "prediction": "ERROR",
        "confidence": 0.0,
        "reasoning": ""
    }
    


# =====================================================
# STAGE 1
# SHARED RAG RETRIEVAL
# =====================================================

def stage1_shared_rag(
    marker_list,
    original_prompt
):

    rag_output = build_rag_prompt(
        marker_list=marker_list,
        original_prompt=original_prompt,
        top_k=4
    )

    print("\n===== RETRIEVAL PACKET =====")
    print(rag_output["retrieval_packet"])
    print("============================\n")

    return rag_output


# =====================================================
# STAGE 2
# INDIVIDUAL COUNCIL MEMBERS
# =====================================================

async def stage2_collect_responses(
    rag_prompt: str
):

    results = []

    for model_name in COUNCIL_MEMBERS:

        start = time.perf_counter()

        response = await ModelRouter.run_async(
            model_name,
            rag_prompt
        )

        elapsed = time.perf_counter() - start

        print(f"[TIMING] {model_name}: {elapsed:.2f} sec")

        results.append(response)

    council_outputs = []

    for model_name, response in zip(
        COUNCIL_MEMBERS,
        results
    ):

        parsed = safe_json_parse(response)

        council_outputs.append(
            {
                "model": model_name,
                "prediction": parsed.get(
                    "prediction",
                    "ERROR"
                ),
                "confidence": parsed.get(
                    "confidence",
                    0.0
                ),
                "reasoning": parsed.get(
                    "reasoning",
                    ""
                ),
                "raw_response": response
            }
        )

    print("\n===== COUNCIL OUTPUTS =====")

    for item in council_outputs:
        print(item)

    print("===========================\n")

    return council_outputs


# =====================================================
# STAGE 3
# CHAIRMAN SYNTHESIS
# =====================================================

async def stage3_chairman(
    marker_list,
    retrieval_packet,
    council_outputs
):

    council_block = "\n\n".join(
        f"""
MODEL: {x['model']}
PREDICTION: {x['prediction']}
CONFIDENCE: {x['confidence']}
REASONING: {x['reasoning']}
""".strip()
        for x in council_outputs
    )

    chairman_prompt = f"""
You are the chairman of a biomedical AI council responsible for producing the final consensus annotation for a single-cell RNA sequencing (scRNA-seq) sample.

Your role is to synthesize the council's analysis rather than replace it. If the council's conclusions are not supported by the biological evidence, you may reach a different conclusion, but you must explicitly justify why.

Evaluate:

1. The marker genes.
2. The retrieved CellMarker evidence.
3. Every council member's prediction.
4. Every council member's biological reasoning.
5. Every council member's confidence score.

**Marker Genes:**

{marker_list}

**Retrieved Evidence:**

{json.dumps(retrieval_packet, indent=2)}

The retrieved candidates are supporting evidence, **not** ground truth.

Interpret the retrieval as follows:

* Higher weighted similarity indicates stronger biological agreement.
* Shared marker genes explain why a candidate was retrieved.
* Evidence count reflects supporting literature only and is **not** a confidence score.

**Council Outputs:**

{council_block}

**Decision process**

1. Assess the marker genes independently.
2. Evaluate the retrieved evidence.
3. Compare the biological reasoning from every council member.
4. Determine whether there is a genuine majority opinion.
5. Decide whether that majority is biologically justified.
6. Produce the final consensus prediction.

**Marker gene interpretation**

* Marker genes are presented in arbitrary order and have no implied ranking or grouping.
* The order of genes should not influence your reasoning.
* Some marker sets may not contain many classic lineage-defining markers. In such cases, evaluate the complete marker profile collectively rather than relying only on a few highly specific genes.
* Consider all provided marker genes during your biological assessment, while emphasizing only the most informative markers in your final explanation.

**Rules**

* Do **not** automatically follow the majority.
* Do **not** automatically follow the retrieved evidence.
* Judge arguments by their biological quality rather than confidence scores.
* If all council members agree, only disagree when the biological evidence clearly contradicts their conclusion.
* If you reject either the majority or the retrieved evidence, explain why.
* Your reasoning NEEDS to include the council member outputs for justification.

Your reasoning must include four sections:

**1. Retrieval Assessment**

* Briefly evaluate whether the retrieved candidates are biologically convincing.

**2. Council Evaluation**

* Summarize each council member's prediction.
* Briefly explain whether each model's reasoning was biologically convincing.

**3. Chairman Assessment**

* Using your own biological knowledge, identify the cell type best supported by the marker genes and explain why.

**4. Final Consensus**

* Explain how the retrieval evidence, council discussion, and your own biological assessment produced the final prediction.

Return **ONLY** valid JSON.

{{
"prediction": "<final cell type>",
"confidence": 0.00,
"reasoning": "Provide a detailed consensus explanation (4–8 sentences) following the required four-section structure."
}}

Do not output markdown.
Do not output code fences.
Do not output any text before the JSON.
Do not output any text after the JSON.
""".strip()

    print("\n===== CHAIRMAN PROMPT =====")
    print(chairman_prompt[:5000])
    print("===========================\n")

    start = time.perf_counter()

    response = await ModelRouter.run_async(
        CHAIRMAN_MODEL,
        chairman_prompt
    )

    print(
        f"[TIMING] Chairman ({CHAIRMAN_MODEL}): "
        f"{time.perf_counter() - start:.2f} sec"
    )

    print("\n===== CHAIRMAN RAW RESPONSE =====")
    print(response)
    print("=================================\n")

    return (
        safe_json_parse(response),
        response
    )


# =====================================================
# STAGE 4
# FULL COUNCIL PIPELINE
# =====================================================

async def run_full_council(
    marker_list,
    original_prompt
):

    # ---------------------------------
    # STAGE 1
    # ---------------------------------

    rag_start = time.perf_counter()

    rag_output = stage1_shared_rag(
        marker_list=marker_list,
        original_prompt=original_prompt
    )

    print(f"[TIMING] RAG: {time.perf_counter() - rag_start:.2f} sec")

    rag_prompt = rag_output["prompt"]

    print("\n===== FINAL RAG PROMPT =====")
    print(rag_prompt[:5000])
    print("============================\n")

    retrieval_packet = rag_output["retrieval_packet"]

    # ---------------------------------
    # STAGE 2
    # ---------------------------------

    council_outputs = await stage2_collect_responses(
        rag_prompt
    )

    # ---------------------------------
    # STAGE 3
    # ---------------------------------

    chairman_output, chairman_raw = await stage3_chairman(
        marker_list,
        retrieval_packet,
        council_outputs
    )
    initial_prediction = ""

    if council_outputs:
        initial_prediction = (
            (council_outputs[0].get("prediction") or "")
            .strip()
            .lower()
        )

    final_prediction = (
        (chairman_output.get("prediction") or "")
        .strip()
        .lower()
    )

    chairman_output["changed_prediction"] = (
        initial_prediction != final_prediction
    )

    # ---------------------------------
    # Majority vote
    # ---------------------------------

    vote_counts = {}

    for member in council_outputs:

        pred = (
            (member.get("prediction") or "")
            .strip()
            .lower()
        )

        vote_counts[pred] = vote_counts.get(pred, 0) + 1

    majority_prediction = None

    majority_size = max(vote_counts.values()) if vote_counts else 0

    # Only 2 or more identical votes count
    # as a true majority.

    if majority_size >= 2:

        for prediction, count in vote_counts.items():

            if count == majority_size:

                majority_prediction = prediction
                break

    chairman_prediction_norm = (
        (chairman_output.get("prediction") or "")
        .strip()
        .lower()
    )

    if majority_prediction is None:

        chairman_followed_majority = "No Majority"

    else:

        chairman_followed_majority = (
            chairman_prediction_norm
            ==
            majority_prediction
        )
    # ---------------------------------
    # RAG Top-4
    # ---------------------------------

    rag_top = [
        c["cell_name"]
        for c in retrieval_packet["top_candidates"]
    ]

    while len(rag_top) < 4:
        rag_top.append("")

    chairman_followed_any_rag = False

    chairman_followed_any_rag = any(
        candidate
        and chairman_prediction_norm == candidate.lower().strip()
        for candidate in rag_top
    )

    for candidate in rag_top:

        if (
            candidate
            and
            chairman_prediction_norm
            ==
            candidate.lower().strip()
        ):
            chairman_followed_any_rag = True
            break

    # ---------------------------------
    # Metadata
    # ---------------------------------

    metadata = {

        "rag_enabled": True,

        "top_k": 4,

        "models_used": COUNCIL_MEMBERS,

        "chairman": CHAIRMAN_MODEL,

        "rag_top1": rag_top[0],
        "rag_top2": rag_top[1],
        "rag_top3": rag_top[2],
        "rag_top4": rag_top[3],

        "majority_prediction": majority_prediction,

        "chairman_followed_majority":
            chairman_followed_majority,

        "chairman_followed_any_rag":
            chairman_followed_any_rag,
            }

    return (
        council_outputs,
        chairman_output,
        chairman_raw,
        metadata
    )