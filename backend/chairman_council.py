"""
Chairman-Centric RAG Council
NTU Cell Annotation Project

PIPELINE

Stage 1:
    Shared RAG Retrieval
    +
    Chairman Preliminary Hypothesis

Stage 2:
    Council Review of Chairman Hypothesis

Stage 3:
    Chairman Final Consensus

Stage 4:
    Structured Return
"""

from typing import Dict
import json
import time
import re

from .config import (
    COUNCIL_MEMBERS,
    CHAIRMAN_MODEL
)

from .model_router import ModelRouter

# Chairman-centric prompt builder
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

    text = str(text).strip()

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
    # Total failure
    # --------------------------------------------------

    print("\n========== JSON PARSE FAILED ==========")
    print(text)
    print("=======================================\n")

    return {

        "prediction": "ERROR",

        "confidence": 0.0,

        "reasoning": text
    }
# =====================================================
# STAGE 1
# SHARED RAG + PRELIMINARY CHAIRMAN HYPOTHESIS
# =====================================================

async def stage1_shared_rag_and_chairman(
    marker_list,
    original_prompt
):
    """
    Stage 1

    1. Perform shared RAG retrieval.
    2. Give ONLY the chairman access to:
            • marker genes
            • retrieved candidates
    3. Produce an INITIAL biological hypothesis.

    This hypothesis is NOT the final decision.
    It is intended to be reviewed and challenged
    by the council in Stage 2.
    """

    # -------------------------------------------------
    # Shared retrieval
    # -------------------------------------------------

    rag_output = build_rag_prompt(
        marker_list=marker_list,
        original_prompt=original_prompt,
        top_k=4
    )

    retrieval_packet = rag_output["retrieval_packet"]

    print("\n===== RETRIEVAL PACKET =====")
    print(retrieval_packet)
    print("============================\n")

    # -------------------------------------------------
    # Preliminary Chairman Prompt
    # -------------------------------------------------

    chairman_prompt = f"""
You are the chairman of a biomedical AI council performing single-cell RNA sequencing (scRNA-seq) cell type annotation.

This is your INITIAL assessment only.

Your role is to produce the strongest biological hypothesis based on the evidence. The council will independently review and may challenge your conclusion.

## Marker Genes

{marker_list}

## Retrieved Evidence

{json.dumps(retrieval_packet, indent=2)}

The retrieved candidates are supporting evidence only.

• Higher similarity suggests stronger biological similarity.
• Retrieved candidates may be incorrect.
• Base your conclusion primarily on the marker genes.

## Instructions

1. Independently interpret the complete marker profile.
2. Identify the most biologically informative lineage markers.
3. Determine the cell type best supported by the evidence.
4. Acknowledge any biological uncertainty.
5. Suggest plausible alternative cell types if appropriate.

## Guidelines

• Marker order has NO biological meaning.
• Evaluate the COMPLETE marker profile.
• Do not rely on only one or two markers.
• Do not automatically trust the retrieval.
• Confidence should reflect biological evidence only.
• Mention ONLY the 3–4 most informative marker genes.
• Do NOT list large numbers of marker genes.
• Keep all text concise to minimize context length.
• Avoid repetition.

## Output Requirements

Return EXACTLY ONE valid JSON object.

Every field below MUST be present.

The prediction field MUST contain the actual biological cell type you conclude is correct.

Never output placeholder text such as:
<cell type>
<most likely cell type>

Do NOT output:
- markdown
- code fences
- explanations
- multiple JSON objects
- any text before or after the JSON

Return exactly this schema:

{{
    "prediction":"Memory B cell",
    "confidence":0.00,
    "rationale":"Concise biological reasoning in 3–4 sentences.",
    "key_markers":[
        "...",
        "...",
        "...",
        "..."
    ],
    "uncertainties":[
        "...",
        "..."
    ],
    "alternative_hypotheses":[
        "...",
        "..."
    ]
}}

REMEMBER:
Output ONE and ONLY ONE JSON object.
All fields are REQUIRED.
After the closing "}}", stop generating immediately.
""".strip()

    print("\n===== PRELIMINARY CHAIRMAN PROMPT =====")
    print(chairman_prompt[:5000])
    print("=======================================\n")

    start = time.perf_counter()

    response = await ModelRouter.run_async(
        CHAIRMAN_MODEL,
        chairman_prompt
    )

    elapsed = time.perf_counter() - start

    print(
        f"[TIMING] Preliminary Chairman ({CHAIRMAN_MODEL}): "
        f"{elapsed:.2f} sec"
    )

    print("\n===== PRELIMINARY CHAIRMAN RAW OUTPUT =====")
    print(response)
    print("===========================================\n")

    chairman_initial_raw = response

    chairman_initial = safe_json_parse(response)

    return {
        "retrieval_packet": retrieval_packet,
        "rag_prompt": rag_output["prompt"],
        "chairman_initial": chairman_initial,
        "chairman_initial_raw": chairman_initial_raw
    }

# =====================================================
# STAGE 2
# COUNCIL REVIEW OF CHAIRMAN HYPOTHESIS
# =====================================================

async def stage2_council_review(
    marker_list,
    rag_prompt,
    chairman_initial
):
    """
    Stage 2

    The chairman has already produced an initial hypothesis.

    Every council member independently reviews that hypothesis.

    Their job is NOT to simply agree or disagree.

    Their job is to critically evaluate the biological evidence
    and determine whether the chairman's hypothesis is justified.
    """

    results = []

    # -------------------------------------------------
    # Chairman summary
    # -------------------------------------------------

    chairman_block = f"""
PRELIMINARY CHAIRMAN HYPOTHESIS

Prediction:
{chairman_initial.get("prediction","ERROR")}

Confidence:
{chairman_initial.get("confidence",0)}

Rationale:
{chairman_initial.get("rationale","")}

Key markers:
{chairman_initial.get("key_markers",[])}

Uncertainties:
{chairman_initial.get("uncertainties",[])}

Alternative hypotheses:
{chairman_initial.get("alternative_hypotheses",[])}
""".strip()

    # -------------------------------------------------
    # Run every council member
    # -------------------------------------------------

    for model_name in COUNCIL_MEMBERS:

        council_prompt = f"""You are an expert computational biologist reviewing a proposed cell-type annotation for a single-cell RNA sequencing (scRNA-seq) sample.

The chairman has already proposed an INITIAL biological hypothesis.

Your task is to independently evaluate the biological evidence and determine whether the chairman's conclusion is justified.

==================================================

MARKER GENES

{marker_list}

==================================================

RETRIEVED EVIDENCE

{rag_prompt}

The retrieved candidates are supporting evidence only.

• Higher similarity scores indicate stronger biological similarity.
• Retrieved candidates are not guaranteed to be correct.
• Use the marker genes as your primary source of evidence.
• If the marker genes support a different cell type, you should disagree with both the retrieval and the chairman when appropriate.

==================================================

CHAIRMAN'S INITIAL HYPOTHESIS

{chairman_block}

==================================================

YOUR REVIEW

Perform an independent biological review.

1. Interpret the marker genes before considering the chairman's conclusion.
2. Identify the strongest lineage-defining markers.
3. Determine the biological cell type best supported by the evidence.
4. Compare your conclusion with the chairman's hypothesis.
5. Decide whether you agree or disagree.
6. Explain your conclusion using biological evidence rather than confidence scores or retrieval ranking.

==================================================

IMPORTANT

Your highest priority is to determine the correct biological cell type.

Before thinking about formatting, determine:

• the best-supported prediction
• the biological reasoning behind that prediction

The **prediction** and **reasoning** are the two most important fields and should always be completed.

The remaining fields simply summarize your review.

==================================================

GUIDELINES

• Marker order has no biological meaning.
• Evaluate the complete marker profile.
• Do not rely on only one or two marker genes.
• Do not automatically trust the chairman.
• Do not automatically trust the retrieved candidates.
• Confidence should reflect only the biological evidence.
• Mention only the 3–5 most informative marker genes.
• Do not list every marker.
• Keep the reasoning concise (2–4 sentences).

==================================================

OUTPUT

Return a single JSON object using this structure.

{{
"prediction":"Memory B cell",
"confidence":0.00,
"agree_with_chairman":true,
"supporting_evidence":"...",
"contradicting_evidence":"None",
"reasoning":"Concise biological justification."
}}

Please complete every field.

If there is no contradicting evidence, write:

"contradicting_evidence":"None"

Return only the JSON object.

Do not include markdown, code fences, or additional commentary.

After the final closing brace, stop generating.
""".strip()

        start = time.perf_counter()

        print("\n===== COUNCIL PROMPT =====")
        print(f"MODEL: {model_name}")
        print(council_prompt)
        print("==========================\n")

        response = await ModelRouter.run_async(
            model_name,
            council_prompt
        )

        print(f"\n===== RAW OUTPUT ({model_name}) =====")
        print(repr(response))
        print("=====================================\n")

        elapsed = time.perf_counter() - start

        print(
            f"[TIMING] {model_name}: "
            f"{elapsed:.2f} sec"
        )

        results.append(response)

    # -------------------------------------------------
    # Parse outputs
    # -------------------------------------------------

    council_reviews = []

    for model_name, response in zip(
        COUNCIL_MEMBERS,
        results
    ):

        parsed = safe_json_parse(response)

        council_reviews.append({

            "model":model_name,

            "prediction":
                parsed.get(
                    "prediction",
                    "ERROR"
                ),

            "confidence":
                parsed.get(
                    "confidence",
                    0.0
                ),

            "agree_with_chairman":
                parsed.get(
                    "agree_with_chairman",
                    False
                ),

            "supporting_evidence":
                parsed.get(
                    "supporting_evidence",
                    ""
                ),

            "contradicting_evidence":
                parsed.get(
                    "contradicting_evidence",
                    ""
                ),

            "reasoning":
                parsed.get(
                    "reasoning",
                    ""
                ),

            "raw_response":
                response
        })

    print("\n===== COUNCIL REVIEWS =====")

    for review in council_reviews:
        print(review)

    print("===========================\n")

    return council_reviews

# =====================================================
# STAGE 3
# CHAIRMAN FINAL CONSENSUS
# =====================================================

async def stage3_final_chairman(
    marker_list,
    retrieval_packet,
    chairman_initial,
    council_reviews
):
    """
    Stage 3

    The chairman now performs a SECOND PASS.

    It reviews:

    • its original hypothesis
    • every council review
    • retrieved evidence
    • marker genes

    and decides whether its original conclusion should
    be maintained or revised.
    """

    # -------------------------------------------------
    # Chairman Initial Hypothesis
    # -------------------------------------------------

    chairman_initial_block = f"""
Prediction:
{chairman_initial.get("prediction","ERROR")}

Confidence:
{chairman_initial.get("confidence",0)}

Rationale:
{chairman_initial.get("rationale","")}

Key Markers:
{chairman_initial.get("key_markers",[])}

Uncertainties:
{chairman_initial.get("uncertainties",[])}

Alternative Hypotheses:
{chairman_initial.get("alternative_hypotheses",[])}
""".strip()

    # -------------------------------------------------
    # Council Reviews
    # -------------------------------------------------

    council_block = "\n\n".join(

f"""
MODEL:
{x["model"]}

Prediction:
{x["prediction"]}

Confidence:
{x["confidence"]}

Agrees with Chairman:
{x["agree_with_chairman"]}

Supporting Evidence:
{x["supporting_evidence"]}

Contradicting Evidence:
{x["contradicting_evidence"]}

Independent Reasoning:
{x["reasoning"]}
""".strip()

        for x in council_reviews
    )

    # -------------------------------------------------
    # Final Chairman Prompt
    # -------------------------------------------------

    chairman_prompt = f"""
You are the chairman of a biomedical AI council performing single-cell RNA sequencing (scRNA-seq) cell type annotation.

Earlier, you produced an INITIAL biological hypothesis.

The council has now independently reviewed your hypothesis.

Your task is to produce the FINAL consensus using the complete biological evidence.

==================================================

MARKER GENES

{marker_list}

==================================================

RETRIEVED EVIDENCE

{json.dumps(retrieval_packet, indent=2)}

Treat the retrieved candidates as supporting evidence only.
Higher similarity scores indicate stronger biological similarity but do not guarantee correctness.

==================================================

YOUR INITIAL HYPOTHESIS

{chairman_initial_block}

==================================================

COUNCIL REVIEWS

{council_block}

==================================================

TASK

1. Re-evaluate your original hypothesis.
2. Consider the council reviews and retrieved evidence.
3. Decide whether your original prediction should be maintained or changed.
4. Base your final decision only on the biological evidence, not confidence scores, authority, retrieval ranking or majority vote.

==================================================

GUIDELINES

• Marker order has no biological meaning.
• Evaluate the complete marker profile.
• Mention only the 3–5 most informative marker genes.
• Do not list large numbers of markers.
• Keep your reasoning concise (3–5 sentences).
• Keep "reason_for_change" to a single sentence.

==================================================

Return one valid JSON object containing ALL of these fields.

The "prediction" field must contain the final biological cell type.

The "reasoning" field is REQUIRED and must never be omitted.

Return this exact structure:

{{
    "prediction":"Memory B cell",
    "confidence":0.00,
    "changed_prediction":false,
    "reason_for_change":"One sentence explaining why the prediction was maintained or changed.",
    "reasoning":"Concise biological justification."
}}

Return only the JSON object.

Do not return multiple JSON objects.

After the final closing brace (}}), stop generating immediately.
""".strip()

    print("\n===== FINAL CHAIRMAN PROMPT =====")
    print(chairman_prompt[:5000])
    print("=================================\n")

    start = time.perf_counter()

    response = await ModelRouter.run_async(
        CHAIRMAN_MODEL,
        chairman_prompt
    )

    print(
        f"[TIMING] Final Chairman ({CHAIRMAN_MODEL}): "
        f"{time.perf_counter()-start:.2f} sec"
    )

    print("\n===== FINAL CHAIRMAN RAW OUTPUT =====")
    print(response)
    print("=====================================\n")

    chairman_final_raw = response

    chairman_final = safe_json_parse(response)

    return (
        chairman_final,
        chairman_final_raw
    )

# =====================================================
# STAGE 4
# FULL CHAIRMAN-CENTRIC PIPELINE
# =====================================================

async def run_full_council(
    marker_list,
    original_prompt
):

    # ---------------------------------
    # STAGE 1
    # Shared RAG + Chairman Hypothesis
    # ---------------------------------

    stage1_start = time.perf_counter()

    stage1_output = await stage1_shared_rag_and_chairman(
        marker_list=marker_list,
        original_prompt=original_prompt
    )

    print(
        f"[TIMING] Stage 1: "
        f"{time.perf_counter()-stage1_start:.2f} sec"
    )

    retrieval_packet = stage1_output["retrieval_packet"]

    rag_prompt = stage1_output["rag_prompt"]

    chairman_initial = stage1_output["chairman_initial"]

    chairman_initial_raw = stage1_output["chairman_initial_raw"]
    # ---------------------------------
    # STAGE 2
    # Council Review
    # ---------------------------------

    stage2_start = time.perf_counter()

    council_reviews = await stage2_council_review(
        marker_list=marker_list,
        rag_prompt=rag_prompt,
        chairman_initial=chairman_initial
    )

    print(
        f"[TIMING] Stage 2: "
        f"{time.perf_counter()-stage2_start:.2f} sec"
    )

    # ---------------------------------
    # STAGE 3
    # Final Chairman Consensus
    # ---------------------------------

    stage3_start = time.perf_counter()

    (
        chairman_final,
        chairman_final_raw
    ) = await stage3_final_chairman(
        marker_list=marker_list,
        retrieval_packet=retrieval_packet,
        chairman_initial=chairman_initial,
        council_reviews=council_reviews
    )

    print(
        f"[TIMING] Stage 3: "
        f"{time.perf_counter()-stage3_start:.2f} sec"
    )

    # ---------------------------------
    # Did the chairman change prediction?
    # ---------------------------------

    initial_prediction = (
        (chairman_initial.get("prediction") or "")
        .strip()
        .lower()
    )

    final_prediction = (
        (chairman_final.get("prediction") or "")
        .strip()
        .lower()
    )

    chairman_final["changed_prediction"] = (
        initial_prediction != final_prediction
    )

    # ---------------------------------
    # Majority Vote
    # ---------------------------------

    vote_counts = {}

    for member in council_reviews:

        pred = (
            (member.get("prediction") or "")
            .strip()
            .lower()
        )

        vote_counts[pred] = vote_counts.get(pred, 0) + 1

    majority_prediction = None

    majority_size = max(vote_counts.values())

    if majority_size >= 2:

        for prediction, count in vote_counts.items():

            if count == majority_size:

                majority_prediction = prediction
                break

    chairman_prediction_norm = (
        (chairman_final.get("prediction") or "")
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
    # RAG Tracking
    # ---------------------------------

    rag_top = [
        c["cell_name"]
        for c in retrieval_packet["top_candidates"]
    ]

    while len(rag_top) < 4:
        rag_top.append("")


    chairman_followed_any_rag = False

    # ---------------------------------
    # Did the chairman follow ANY retrieved candidate?
    # ---------------------------------

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

        "architecture": "chairman_centric",

        "rag_enabled": True,

        "top_k": 4,

        "chairman": CHAIRMAN_MODEL,

        "models_used": COUNCIL_MEMBERS,

        "rag_top1": rag_top[0],

        "rag_top2": rag_top[1],

        "rag_top3": rag_top[2],

        "rag_top4": rag_top[3],

        "majority_prediction": majority_prediction,

        "chairman_followed_majority":
            chairman_followed_majority,

        "chairman_followed_any_rag":
            chairman_followed_any_rag

    }

    return (

        council_reviews,

        chairman_initial,
        chairman_initial_raw,

        chairman_final,
        chairman_final_raw,

        metadata

    )