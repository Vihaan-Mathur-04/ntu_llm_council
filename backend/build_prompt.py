def build_rag_context(packet):
    """
    Convert a retrieval evidence packet into
    a text block suitable for insertion into
    an LLM prompt.

    Input:
        build_evidence_packet()

    Output:
        String
    """

    lines = []

    lines.append(
        "=== RETRIEVED CELL TYPE EVIDENCE ==="
    )

    lines.append("")

    lines.append(
        "Observed Marker Genes:"
    )

    lines.append(
        ", ".join(packet["query_markers"])
    )

    lines.append("")

    lines.append(
        "Top Retrieved Candidate Cell Types:"
    )

    lines.append("")

    for candidate in packet[
        "top_candidates"
    ]:

        lines.append(
            f"{candidate['rank']}. "
            f"{candidate['cell_name']}"
        )

        lines.append(
            f"   Overlap Score: "
            f"{candidate['overlap_score']}"
        )

        lines.append(
            f"   Shared Markers: "
            + ", ".join(
                candidate[
                    "shared_markers"
                ]
            )
        )

        lines.append(
            f"   Evidence Count: "
            f"{candidate['evidence_count']}"
        )

        lines.append("")

    lines.append(
        "Use the retrieved evidence above "
        "to support your reasoning."
    )

    lines.append(
        "Do not simply choose the top-ranked "
        "candidate. Evaluate all evidence "
        "carefully before making a final "
        "cell type prediction."
    )

    return "\n".join(lines)