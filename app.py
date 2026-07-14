"""
app.py
Minimal Flask server with a single /chat endpoint.
Flow: user question -> retrieve relevant FAQ chunks from ChromaDB ->
build a prompt with those chunks -> ask the LLM -> return the answer.

LLM provider: Groq (free, no card needed). Get a free key at
https://console.groq.com and set it as the GROQ_API_KEY environment variable.

To switch to Anthropic's Claude later (once you have API credits), see the
commented-out block in ask_llm() below — it's a drop-in swap.
"""

import os
from dotenv import load_dotenv; 
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
from rag_core import retrieve


app = Flask(__name__, static_folder="static")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = (
    "You are a helpful assistant for a dental clinic's website. "
    "Answer the user's question using ONLY the FAQ context provided. "
    "If the answer isn't in the context, say you don't have that "
    "information and suggest they call the clinic directly. "
    "Keep answers short, friendly, and in plain language."
)


def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)
    user_msg = f"FAQ context:\n{context}\n\nQuestion: {question}"

    if not groq_client:
        return (
            "[Demo mode: no GROQ_API_KEY set] Based on the FAQ, here's the "
            f"most relevant snippet:\n\n{context_chunks[0]}"
        )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content

    # --- To use Claude instead, once you have API credits ---
    # import anthropic
    # client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=300,
    #     system=SYSTEM_PROMPT,
    #     messages=[{"role": "user", "content": user_msg}],
    # )
    # return response.content[0].text


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = (data or {}).get("message", "").strip()
    if not question:
        return jsonify({"error": "message field is required"}), 400

    chunks = retrieve(question, k=3)
    answer = ask_llm(question, chunks)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)