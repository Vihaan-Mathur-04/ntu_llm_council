import json

CORPUS_FILE = "cellmarker_corpus.json"
TOP_K = 10


def load_corpus():
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_by_overlap(query_markers, corpus, top_k=10):
    """
    Retrieval V2

    Primary score:
        Number of shared markers

    Secondary score:
        Fraction of cell markers matched

    Tertiary score:
        Evidence count
    """

    query_set = {
        marker.upper().strip()
        for marker in query_markers
    }

    results = []

    for entry in corpus:

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

        overlap_fraction = overlap_score / len(cell_markers)

        results.append(
            {
                "cell_name": entry["cell_name"],
                "score": overlap_score,
                "overlap_fraction": overlap_fraction,
                "marker_count": entry["marker_count"],
                "evidence_count": entry["evidence_count"],
                "shared_markers": sorted(shared)
            }
        )

    results.sort(
        key=lambda x: (
            x["score"],
            x["overlap_fraction"],
            x["evidence_count"]
        ),
        reverse=True
    )

    return results[:top_k]


def print_results(query_markers, results):

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(", ".join(query_markers))

    print("\n" + "=" * 80)
    print("TOP RESULTS")
    print("=" * 80)

    if not results:
        print("No matches found.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\n#{rank}")
        print("-" * 40)

        print(f"Cell Type         : {result['cell_name']}")
        print(f"Overlap Score     : {result['score']}")
        print(f"Overlap Fraction  : {result['overlap_fraction']:.4f}")
        print(f"Marker Count      : {result['marker_count']}")
        print(f"Evidence Count    : {result['evidence_count']}")

        print(
            "Shared Markers    : "
            + ", ".join(result["shared_markers"])
        )


def run_test(query_markers):

    corpus = load_corpus()

    results = retrieve_by_overlap(
        query_markers=query_markers,
        corpus=corpus,
        top_k=TOP_K
    )

    print_results(query_markers, results)


if __name__ == "__main__":

    TEST_QUERIES = {

        "B_CELL": [
            "MS4A1",
            "CD79A",
            "CD79B"
        ],

        "T_CELL": [
            "CD3D",
            "CD3E",
            "TRBC1"
        ],

        "NK_CELL": [
            "NKG7",
            "GNLY",
            "KLRD1"
        ],

        "MONOCYTE": [
            "LYZ",
            "FCN1",
            "S100A8"
        ]
    }

    for name, markers in TEST_QUERIES.items():

        print("\n")
        print("#" * 80)
        print(name)
        print("#" * 80)

        run_test(markers)