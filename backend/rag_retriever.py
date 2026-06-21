import json


CORPUS_FILE = "backend\\cellmarker_corpus.json"

DEFAULT_TOP_K = 10


def load_corpus():

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


CORPUS = load_corpus()


def retrieve_candidates(
    query_markers,
    top_k=DEFAULT_TOP_K
):
    """
    Retrieve candidate cell types using
    marker overlap.

    Returns:
        List[dict]
    """

    query_set = {
        marker.upper().strip()
        for marker in query_markers
    }

    results = []

    for entry in CORPUS:

        cell_markers = {
            marker.upper().strip()
            for marker in entry["markers"]
        }

        if not cell_markers:
            continue

        shared = query_set & cell_markers

        overlap_score = len(shared)

        if overlap_score == 0:
            continue

        overlap_fraction = (
            overlap_score /
            len(cell_markers)
        )

        results.append(
            {
                "cell_name":
                    entry["cell_name"],

                "overlap_score":
                    overlap_score,

                "overlap_fraction":
                    round(
                        overlap_fraction,
                        4
                    ),

                "marker_count":
                    entry["marker_count"],

                "evidence_count":
                    entry["evidence_count"],

                "shared_markers":
                    sorted(shared),

                "all_markers":
                    entry["markers"]
            }
        )

    results.sort(
        key=lambda x: (
            x["overlap_score"],
            x["overlap_fraction"],
            x["evidence_count"]
        ),
        reverse=True
    )

    return results[:top_k]


def build_evidence_packet(
    query_markers,
    top_k=DEFAULT_TOP_K
):
    """
    Main RAG retrieval function.

    Returns a structured evidence packet
    for council consumption.
    """

    candidates = retrieve_candidates(
        query_markers=query_markers,
        top_k=top_k
    )

    packet = {

        "query_markers":
            query_markers,

        "candidate_count":
            len(candidates),

        "top_candidates":
            []
    }

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        packet[
            "top_candidates"
        ].append(
            {
                "rank":
                    rank,

                "cell_name":
                    candidate["cell_name"],

                "overlap_score":
                    candidate["overlap_score"],

                "overlap_fraction":
                    candidate["overlap_fraction"],

                "marker_count":
                    candidate["marker_count"],

                "evidence_count":
                    candidate["evidence_count"],

                "shared_markers":
                    candidate["shared_markers"]
            }
        )

    return packet