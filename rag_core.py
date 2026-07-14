"""
rag_core.py
Loads the FAQ doc, splits it into Q&A chunks, embeds them into a local
ChromaDB collection, and exposes a retrieve() function used by app.py.

Embeddings are generated locally with sentence-transformers (free, no API
key needed). Only the final answer-generation step calls an LLM API.
"""

import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = os.path.join(os.path.dirname(__file__), "faq.md")

def load_chunks():
    text = open(FAQ_PATH, "r", encoding="utf-8").read()
    raw_chunks = re.split(r"\n(?=Q:)", text.strip())
    return [c.strip() for c in raw_chunks if c.strip()]

chunks = load_chunks()
vectorizer = TfidfVectorizer().fit(chunks)
chunk_vectors = vectorizer.transform(chunks)

def retrieve(query, k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, chunk_vectors)[0]
    top_k = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k]