"""
rag_core.py
Loads the FAQ doc, splits it into Q&A chunks, embeds them into a local
ChromaDB collection, and exposes a retrieve() function used by app.py.

Embeddings are generated locally with sentence-transformers (free, no API
key needed). Only the final answer-generation step calls an LLM API.
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions

FAQ_PATH = os.path.join(os.path.dirname(__file__), "faq.md")
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Free local embedding model — no API key, no cost, runs on CPU
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_PATH)


def load_chunks(path=FAQ_PATH):
    """Split the FAQ markdown into one chunk per Q&A pair."""
    text = open(path, "r", encoding="utf-8").read()
    # Each entry starts with "Q:" — split on that boundary
    raw_chunks = re.split(r"\n(?=Q:)", text.strip())
    return [c.strip() for c in raw_chunks if c.strip()]


def build_index(force_rebuild=False):
    """Create (or reuse) the ChromaDB collection populated with FAQ chunks."""
    existing = [c.name for c in client.list_collections()]
    if "dental_faq" in existing and not force_rebuild:
        return client.get_collection("dental_faq", embedding_function=embedding_fn)

    if "dental_faq" in existing:
        client.delete_collection("dental_faq")

    collection = client.create_collection("dental_faq", embedding_function=embedding_fn)
    chunks = load_chunks()
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"Indexed {len(chunks)} FAQ chunks into ChromaDB.")
    return collection


def retrieve(query, k=3):
    """Return the top-k most relevant FAQ chunks for a user query."""
    collection = build_index()
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


if __name__ == "__main__":
    # Quick manual test: python rag_core.py
    build_index(force_rebuild=True)
    test_q = "How much does a root canal cost?"
    print(f"\nQuery: {test_q}")
    for chunk in retrieve(test_q):
        print("---")
        print(chunk)