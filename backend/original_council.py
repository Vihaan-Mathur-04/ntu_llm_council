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

def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Extracts JSON even when LLM output is messy.
    """

    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {
        "ranking": [],
        "consensus": "ERROR",
        "conflicts": ["Failed to parse JSON output"],
        "reasoning_summary": text
    }


# =====================================================
# STAGE 1: COUNCIL RESPONSES
# =====================================================

async def stage1_collect_responses(
    user_query: str,
    cellstar_context: Any = None
) -> List[Dict[str, Any]]:

    # NOTE:
    # Still async gather, but SAFE because models are NOT persistent
    tasks = [
        ModelRouter.run_async(model_name, user_query)
        for model_name in COUNCIL_MEMBERS
    ]

    results = await asyncio.gather(*tasks)

    return [
        {
            "model": model_name,
            "response": response,
            "type": "llm"
        }
        for model_name, response in zip(COUNCIL_MEMBERS, results)
        if response is not None
    ]


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
        f"[{r['model']}]\n{r['response']}"
        for r in stage1_results
    )

    chairman_prompt = f"""
You are the chairman of a biomedical AI council.

Inputs:
(1) Raw council reasoning
(2) Stage 2 arbitration summary

QUESTION:
{user_query}

RAW RESPONSES:
{council_block}

TASK:
- Resolve conflicts intelligently
- Use arbitration as guidance, not authority
- Prefer biologically correct interpretation
- Output ONLY final cell type label

Final answer:
"""

    response = await ModelRouter.run_async(
        CHAIRMAN_MODEL,
        chairman_prompt
    )

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.strip() if response else "ERROR"
    }


# =====================================================
# FULL PIPELINE
# =====================================================

async def run_full_council(
    user_query: str,
    cellstar_context: Any = None
) -> Tuple[List, Dict[str, Any], Dict]:

    stage1_results = await stage1_collect_responses(
        user_query,
        cellstar_context
    )

    if not stage1_results:
        return [], {"model": "error", "response": "All models failed."}, {}



    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
    )

    metadata = {
        "num_models": len(stage1_results),
        "models_used": COUNCIL_MEMBERS,
        "stage2_enabled": True,
        "cellstar_used": cellstar_context is not None
    }

    return stage1_results, stage3_result, metadata