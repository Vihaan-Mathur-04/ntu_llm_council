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

from typing import List, Dict, Any, Tuple
import asyncio
import json
import re
from urllib import response

from .config import (
    COUNCIL_MEMBERS,
    CHAIRMAN_MODEL
)

from .model_router import ModelRouter
from .base_centric_rag import build_base_centric_prompt


# =====================================================
# JSON PARSER
# =====================================================

def safe_json_parse(text: str) -> Dict:

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
        "label": "ERROR",
        "confidence": 0,
        "reasoning": text
    }


# =====================================================
# STAGE 1
# SHARED RAG RETRIEVAL
# =====================================================

def stage1_shared_rag(
    marker_list,
    original_prompt
):

    rag_output = build_base_centric_prompt(
        marker_list=marker_list,
        original_prompt=original_prompt,
        top_k=10
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

    tasks = [
        ModelRouter.run_async(
            model_name,
            rag_prompt
        )
        for model_name in COUNCIL_MEMBERS
    ]

    results = await asyncio.gather(*tasks)

    council_outputs = []

    for model_name, response in zip(
        COUNCIL_MEMBERS,
        results
    ):

        council_outputs.append(
            {
                "model": model_name,
                "response": response
            }
        )

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
        f"[{r['model']}]\n{r['response']}"
        for r in council_outputs
    )

    chairman_prompt = f"""
You are the chairman of a biomedical AI council.

Marker genes:
{marker_list}

Retrieved evidence:
{json.dumps(retrieval_packet, indent=2)}

Council outputs:

{council_block}

TASK

1. Review retrieved evidence
2. Review every council response
3. Resolve disagreements
4. Determine final cell type
5. Determine confidence score
6. Provide concise reasoning

Return ONLY valid JSON.

{{
    "label": "...",
    "confidence": 0.0,
    "reasoning": "..."
}}
"""
    print("\n===== CHAIRMAN PROMPT =====")
    print(chairman_prompt[:5000])
    print("==========================\n")

    response = await ModelRouter.run_async(
        CHAIRMAN_MODEL,
        chairman_prompt
    )
    print("===== RESPONSE =====")
    print(response)
    print("=========================================\n")

    return safe_json_parse(response)


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

    rag_output = stage1_shared_rag(
        marker_list=marker_list,
        original_prompt=original_prompt
    )

    rag_prompt = rag_output["prompt"]
    print("\n===== FINAL RAG PROMPT =====")
    print(rag_prompt[:5000])   # don't print entire huge prompt
    print("===========================\n")
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

    chairman_output = await stage3_chairman(
        marker_list,
        retrieval_packet,
        council_outputs
    )

    # ---------------------------------
    # STAGE 4
    # ---------------------------------

    metadata = {
        "rag_enabled": True,
        "top_k": 10,
        "models_used": COUNCIL_MEMBERS,
        "chairman": CHAIRMAN_MODEL
    }

    return (
        council_outputs,
        chairman_output,
        metadata
    )