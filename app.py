# app.py
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate

DB_DIR = "db"
app = Flask(__name__)

# 1) Load the vector store and embeddings (must match ingest)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 2) Local LLM via Ollama
llm = Ollama(model="llama3.1:8b", temperature=0.2)  # swap to "llama3.2:3b" if you want faster

# 3) Retrieval-augmented prompt
PROMPT = PromptTemplate.from_template(
    """You are an RTS Gainesville assistant. Answer accurately and concisely
using ONLY the context below. If the answer is not in the context, say you
don't know and suggest contacting RTS.

Context:
{context}

Question: {question}

Answer:"""
)

def build_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[{i}] (source={src}, page={page})\n{d.page_content}\n")
    return "\n---\n".join(parts)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    k = int(data.get("k", 4))
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # A) retrieve
    docs = vectordb.as_retriever(search_kwargs={"k": k}).get_relevant_documents(question)
    if not docs:
        return jsonify({"answer": "No relevant context found in your documents.", "sources": []})

    # B) build prompt
    context = build_context(docs)
    prompt = PROMPT.format(context=context, question=question)

    # C) generate
    answer = llm.invoke(prompt).strip()

    # D) return with sources
    sources = []
    for d in docs:
        m = d.metadata or {}
        sources.append({"source": m.get("source", "unknown"), "page": m.get("page")})
    return jsonify({"answer": answer, "sources": sources})
@app.route("/", methods=["GET"])
def home():
    return "RAG server is running. Send a POST to /ask with JSON: {question: ..., k: 4}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
