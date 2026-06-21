import json
import numpy as np
import faiss


DOCUMENTS_FILE = "cell_documents.json"
EMBEDDINGS_FILE = "cell_embeddings.npy"

INDEX_FILE = "cell_index.faiss"
METADATA_FILE = "cell_metadata.json"


def load_documents():

    with open(
        DOCUMENTS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_embeddings():

    return np.load(
        EMBEDDINGS_FILE
    ).astype("float32")


def build_index(embeddings):

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    print(
        f"Creating FAISS IndexFlatIP..."
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def save_index(index):

    faiss.write_index(
        index,
        INDEX_FILE
    )


def save_metadata(documents):

    metadata = []

    for i, doc in enumerate(documents):

        metadata.append(
            {
                "id": i,
                "cell_name": doc["cell_name"]
            }
        )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )


def main():

    print("=" * 80)
    print("LOADING DOCUMENTS")
    print("=" * 80)

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents"
    )

    print("\n" + "=" * 80)
    print("LOADING EMBEDDINGS")
    print("=" * 80)

    embeddings = load_embeddings()

    print(
        f"Shape: {embeddings.shape}"
    )

    print("\n" + "=" * 80)
    print("BUILDING INDEX")
    print("=" * 80)

    index = build_index(
        embeddings
    )

    print(
        f"Vectors in index: {index.ntotal}"
    )

    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)

    save_index(index)
    save_metadata(documents)

    print(
        f"Saved index -> {INDEX_FILE}"
    )

    print(
        f"Saved metadata -> {METADATA_FILE}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()