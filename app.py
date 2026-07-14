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

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

#Debug Route

@app.route("/debug")
def debug():
    from rag_core import chunks, retrieve
    test = retrieve("working hours", k=2)
    return jsonify({"total_chunks": len(chunks), "sample_retrieval": test})


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
