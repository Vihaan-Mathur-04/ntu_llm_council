import json


CORPUS_FILE = "/Users/vihaan_mathur/ntu_project/llm_council/backend/cellmarker_corpus.json"

DEFAULT_TOP_K = 10


# =====================================================
# LOAD CORPUS
# =====================================================

def load_corpus():

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


CORPUS = load_corpus()


# =====================================================
# RETRIEVAL
# =====================================================

def retrieve_candidates(
    query_markers,
    top_k=DEFAULT_TOP_K
):
    """
    Retrieve candidate cell types using
    IDF-weighted Dice similarity.
    """

    # ----------------------------------------------------
    # Normalise query markers
    # Accept either a comma-separated string or a list
    # ----------------------------------------------------

    if isinstance(query_markers, str):

        query_markers = [

            gene.strip().upper()

            for gene in query_markers.split(",")

            if gene.strip()

        ]

    else:

        query_markers = [

            str(gene).strip().upper()

            for gene in query_markers

            if str(gene).strip()

        ]

    query_set = set(query_markers)

    results = []

    for entry in CORPUS:

        marker_dict = entry["markers"]

        if not marker_dict:
            continue

        cell_markers = set(marker_dict.keys())

        shared = query_set & cell_markers

        if not shared:
            continue

        # ----------------------------------------
        # Weighted overlap
        # ----------------------------------------

        shared_weight = sum(

            marker_dict[m]["idf"]

            for m in shared

        )

        candidate_weight = sum(

            info["idf"]

            for info in marker_dict.values()

        )

        # Query weight
        #
        # We only know IDF values for markers
        # that exist in the corpus.
        # Unknown markers receive weight = 1.
        # ----------------------------------------

        query_weight = 0

        for marker in query_set:

            if marker in marker_dict:

                query_weight += marker_dict[marker]["idf"]

            else:

                query_weight += 1.0

        weighted_dice = (

            2 * shared_weight

        ) / (

            query_weight +

            candidate_weight

        )

        results.append(

            {

                "cell_name":
                    entry["cell_name"],

                "dice_score":
                    round(weighted_dice, 4),

                "shared_weight":
                    round(shared_weight, 4),

                "overlap_score":
                    len(shared),

                "marker_count":
                    entry["marker_count"],

                "evidence_count":
                    entry["evidence_count"],

                "shared_markers":
                    sorted(shared),

                "all_markers":
                    marker_dict

            }

        )

    results.sort(

        key=lambda x: (

            x["dice_score"],

            x["shared_weight"],

            x["evidence_count"]

        ),

        reverse=True

    )

    return results[:top_k]


# =====================================================
# BUILD EVIDENCE PACKET
# =====================================================

def build_evidence_packet(

    query_markers,

    top_k=DEFAULT_TOP_K

):

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

        packet["top_candidates"].append(

            {

                "rank":
                    rank,

                "cell_name":
                    candidate["cell_name"],

                "dice_score":
                    candidate["dice_score"],

                "shared_weight":
                    candidate["shared_weight"],

                "overlap_score":
                    candidate["overlap_score"],

                "marker_count":
                    candidate["marker_count"],

                "evidence_count":
                    candidate["evidence_count"],

                "shared_markers":
                    candidate["shared_markers"]

            }

        )

    return packet