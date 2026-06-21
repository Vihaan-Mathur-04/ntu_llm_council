from backend.rag_retriever import build_evidence_packet
from backend.build_prompt import build_rag_context


DEFAULT_TOP_K = 10


def build_base_centric_prompt(
    marker_list,
    original_prompt,
    top_k=DEFAULT_TOP_K
):
    """
    Base-Centric RAG

    Retrieval occurs BEFORE
    any council member sees
    the prompt.

    Every model receives the
    same retrieved evidence.
    """

    packet = build_evidence_packet(
        query_markers=marker_list,
        top_k=top_k
    )

    rag_context = build_rag_context(
        packet
    )

    final_prompt = f"""
{original_prompt}

{rag_context}
"""

    return {
        "prompt": final_prompt,
        "retrieval_packet": packet
    }