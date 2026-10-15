"""
3-stage LOCAL LLM Council orchestration (NTU version - LLM-only voting system).
"""

from typing import List, Dict, Any, Tuple
import asyncio
import json
import re

from .config import COUNCIL_MEMBERS, CHAIRMAN_MODEL
from .model_router import ModelRouter


# =====================================================
# SAFE JSON PARSER
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
# STAGE 1: COUNCIL RESPONSES
# =====================================================

async def stage1_collect_responses(
    user_query: str
):

    results = []

    for model_name in COUNCIL_MEMBERS:

        response = await ModelRouter.run_async(
            model_name,
            user_query
        )

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

    return council_outputs


# =====================================================
# STAGE 2: DELIBERATION + RANKING
# =====================================================
#REMOVED FOR NOW
# =====================================================
# STAGE 3: CHAIRMAN SYNTHESIS
# =====================================================

async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    council_block = "\n\n".join(
        f"""
    MODEL: {r['model']}
    PREDICTION: {r['prediction']}
    CONFIDENCE: {r['confidence']}
    REASONING: {r['reasoning']}
    """.strip()
        for r in stage1_results
    )

    chairman_prompt = f"""
You are the chairman of a biomedical AI council responsible for producing the final consensus annotation for a single-cell RNA sequencing (scRNA-seq) sample.

Marker genes:

{user_query}

Council outputs:

{council_block}

Decision process

1. Assess the marker genes independently.
2. Compare every council member's reasoning.
3. Determine whether there is a genuine majority.
4. Decide whether the majority is biologically justified.
5. Produce the final consensus prediction.

Rules

• Do not automatically follow the majority.
• Judge arguments by biological quality.
• If every council member agrees, only disagree when the biological evidence clearly contradicts them.
• Your reasoning must reference the council members. If you mention marker genes in your reasoning only mention 4 - 6 important ones.

Your reasoning must contain

1. Council Evaluation

2. Chairman Assessment

3. Final Consensus

Return ONLY

{{
"prediction":"...",
"confidence":0.00,
"reasoning":"4–8 sentence consensus explanation."
}}

Do not output markdown.

Do not output code fences.

Do not output anything before the JSON.

Do not output anything after the JSON.
"""

    response = await ModelRouter.run_async(
        CHAIRMAN_MODEL,
        chairman_prompt
    )

    parsed = safe_json_parse(response)

    return (
        {
            "model": CHAIRMAN_MODEL,
            "prediction": parsed.get("prediction", "ERROR"),
            "confidence": parsed.get("confidence", 0.0),
            "reasoning": parsed.get("reasoning", ""),
            "raw_response": response
        },
        response
    )



# =====================================================
# FULL PIPELINE
# =====================================================

async def run_full_council(
    user_query: str
) -> Tuple[List, Dict[str, Any], Dict]:

    stage1_results = await stage1_collect_responses(
        user_query
    )

    if not stage1_results:
        return [], {"model": "error", "response": "All models failed."}, {}



    chairman_output, chairman_raw = await stage3_synthesize_final(
        user_query,
        stage1_results,
    )
    # ---------------------------------
    # Majority vote
    # ---------------------------------

    vote_counts = {}

    for member in stage1_results:

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

    metadata = {

        "models_used": COUNCIL_MEMBERS,

        "chairman": CHAIRMAN_MODEL,

        "majority_prediction": majority_prediction,

        "chairman_followed_majority":
            chairman_followed_majority
    }

    return (
        stage1_results,
        chairman_output,
        chairman_raw,
        metadata
    )