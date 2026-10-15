def build_rag_context(packet):
    """
    Convert a retrieval evidence packet into
    a text block suitable for insertion into
    an LLM prompt.
    """

    lines = []

    lines.append("=== RETRIEVED CELL TYPE EVIDENCE ===")
    lines.append("")

    # -----------------------------------------------------
    # Query
    # -----------------------------------------------------

    lines.append("Top Retrieved Candidate Cell Types:")
    lines.append("")

    # -----------------------------------------------------
    # Candidates
    # -----------------------------------------------------

    for candidate in packet["top_candidates"]:

        lines.append(
            f"{candidate['rank']}. {candidate['cell_name']}"
        )

        lines.append(
            f"   Weighted Dice Similarity : {candidate['dice_score']:.3f}"
        )

        lines.append(
            f"   Shared Marker Weight     : {candidate['shared_weight']:.3f}"
        )

        lines.append(
            f"   Shared Marker Count      : {candidate['overlap_score']}"
        )

        lines.append(
            f"   Evidence Count           : {candidate['evidence_count']}"
        )

        lines.append(
            "   Shared Markers           : "
            + ", ".join(candidate["shared_markers"])
        )

        lines.append("")

    # -----------------------------------------------------
    # Instructions
    # -----------------------------------------------------

    lines.append(
        "Interpretation Guidelines:"
    )

    lines.append(
        "- Candidates are ranked using an IDF-weighted Dice similarity score."
    )

    lines.append(
        "- Rare, biologically specific marker genes contribute more than common marker genes."
    )

    lines.append(
        "- Treat the retrieved candidates as supporting biological evidence, not ground truth."
    )

    lines.append(
        "- A lower-ranked candidate may still be correct if its marker profile better explains the observed genes."
    )

    lines.append(
        "- If none of the retrieved candidates adequately explains the marker genes, you may predict a different cell type and explain why."
    )

    return "\n".join(lines)