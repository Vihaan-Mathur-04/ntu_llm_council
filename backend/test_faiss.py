import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

INDEX_FILE = "cell_index.faiss"
METADATA_FILE = "cell_metadata.json"

TOP_K = 10


def load_index():

    return faiss.read_index(
        INDEX_FILE
    )


def load_metadata():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_model():

    print("\nLoading embedding model...")

    return SentenceTransformer(
        MODEL_NAME
    )


def build_query_document(markers):

    marker_text = ", ".join(markers)

    document = f"""
Unknown Cell Type

Known Marker Genes:
{marker_text}
"""

    return document.strip()


def embed_query(
    model,
    markers
):

    query_doc = build_query_document(
        markers
    )

    embedding = model.encode(
        query_doc,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding.astype(
        "float32"
    ).reshape(1, -1)


def search(
    query_embedding,
    index,
    metadata,
    top_k=10
):

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        results.append(
            {
                "cell_name":
                    metadata[idx]["cell_name"],
                "similarity":
                    float(score)
            }
        )

    return results


def print_results(
    query_markers,
    results
):

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(
        ", ".join(query_markers)
    )

    print("\n" + "=" * 80)
    print("FAISS RESULTS")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\n#{rank}")
        print("-" * 40)

        print(
            f"Cell Type : {result['cell_name']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )


def run_query(
    model,
    index,
    metadata,
    markers
):

    query_embedding = embed_query(
        model,
        markers
    )

    results = search(
        query_embedding,
        index,
        metadata,
        TOP_K
    )

    print_results(
        markers,
        results
    )


def main():

    index = load_index()

    metadata = load_metadata()

    model = load_model()

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

        run_query(
            model,
            index,
            metadata,
            markers
        )


if __name__ == "__main__":
    main()