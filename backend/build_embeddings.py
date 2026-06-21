import json
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_FILE = "cellmarker_corpus.json"

DOCUMENTS_FILE = "cell_documents.json"
EMBEDDINGS_FILE = "cell_embeddings.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_corpus():

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document(entry):

    cell_name = entry["cell_name"]

    markers = ", ".join(
        sorted(entry["markers"])
    )

    document = f"""
Cell Type: {cell_name}

Known Marker Genes:
{markers}

Evidence Count:
{entry['evidence_count']}
"""

    return document.strip()


def build_documents(corpus):

    documents = []

    for entry in corpus:

        doc = build_document(entry)

        documents.append(
            {
                "cell_name": entry["cell_name"],
                "document": doc
            }
        )

    return documents


def generate_embeddings(documents):

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [
        doc["document"]
        for doc in documents
    ]

    print(f"Generating embeddings for {len(texts)} cells...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


def save_outputs(documents, embeddings):

    with open(
        DOCUMENTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False
        )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )


def main():

    print("=" * 80)
    print("LOADING CORPUS")
    print("=" * 80)

    corpus = load_corpus()

    print(f"Loaded {len(corpus)} cell types")

    print("\n" + "=" * 80)
    print("BUILDING DOCUMENTS")
    print("=" * 80)

    documents = build_documents(corpus)

    print(
        f"Created {len(documents)} documents"
    )

    print("\n" + "=" * 80)
    print("GENERATING EMBEDDINGS")
    print("=" * 80)

    embeddings = generate_embeddings(
        documents
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)

    save_outputs(
        documents,
        embeddings
    )

    print(
        f"Saved documents -> {DOCUMENTS_FILE}"
    )

    print(
        f"Saved embeddings -> {EMBEDDINGS_FILE}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()