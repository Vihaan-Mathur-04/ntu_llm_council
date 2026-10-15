from backend.rag_retriever import build_evidence_packet
from backend.build_prompt import build_rag_context


DEFAULT_TOP_K = 4


def build_rag_prompt(
    marker_list,
    original_prompt,
    top_k=DEFAULT_TOP_K
):
    """
    Shared RAG Builder

    Constructs a retrieval-augmented prompt by appending
    retrieved CellMarker evidence to an existing prompt.

    The function is architecture-independent and can be
    used by both the base-centric and chairman-centric
    council pipelines.
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