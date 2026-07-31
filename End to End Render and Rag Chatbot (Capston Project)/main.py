import os
import glob
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Chat model: Groq's free tier (no credit card, generous daily limits).
# Get a free key at https://console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Embeddings: run locally via fastembed, completely free, no API key needed.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embedder


CHAT_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CACHE_FILE = BASE_DIR / "embeddings_cache.json"

app = FastAPI(title="Simple RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory vector store: list of {"text": str, "source": str, "embedding": np.array}
VECTOR_STORE = []


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple fixed-size character chunking with overlap."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_texts(texts):
    """Embed locally with fastembed — free, no API key, runs on CPU."""
    embedder = get_embedder()
    return [e.tolist() for e in embedder.embed(texts)]


def build_or_load_index():
    """Chunk + embed everything in docs/. Caches results keyed on file mtimes
    so a redeploy without doc changes doesn't re-call the embeddings API."""
    global VECTOR_STORE

    doc_files = sorted(
        f for f in glob.glob(str(DOCS_DIR / "**/*.*"), recursive=True)
        if f.lower().endswith((".txt", ".md"))
    )
    fingerprint = {f: os.path.getmtime(f) for f in doc_files}

    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if cached.get("fingerprint") == fingerprint:
                VECTOR_STORE = [
                    {"text": c["text"], "source": c["source"], "embedding": np.array(c["embedding"])}
                    for c in cached["chunks"]
                ]
                print(f"Loaded {len(VECTOR_STORE)} chunks from cache.")
                return
        except Exception as e:
            print("Cache load failed, rebuilding:", e)

    print("Building embedding index from docs/ (local model, first run downloads it, may take a minute)...")
    all_chunks, all_sources = [], []
    for f in doc_files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        for c in chunk_text(text):
            all_chunks.append(c)
            all_sources.append(os.path.basename(f))

    if not all_chunks:
        print("No documents found in docs/. Add .txt or .md files there.")
        VECTOR_STORE = []
        return

    embeddings = embed_texts(all_chunks)
    VECTOR_STORE = [
        {"text": t, "source": s, "embedding": np.array(e)}
        for t, s, e in zip(all_chunks, all_sources, embeddings)
    ]

    CACHE_FILE.write_text(json.dumps({
        "fingerprint": fingerprint,
        "chunks": [
            {"text": v["text"], "source": v["source"], "embedding": v["embedding"].tolist()}
            for v in VECTOR_STORE
        ]
    }))
    print(f"Indexed {len(VECTOR_STORE)} chunks.")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def retrieve(query, k=TOP_K):
    if not VECTOR_STORE:
        return []
    q_emb = np.array(embed_texts([query])[0])
    scored = sorted(
        ((cosine_sim(q_emb, item["embedding"]), item) for item in VECTOR_STORE),
        key=lambda x: x[0],
        reverse=True,
    )
    return [item for _, item in scored[:k]]


class ChatRequest(BaseModel):
    message: str
    history: list = []  # [{"role": "user"/"assistant", "content": "..."}]


@app.on_event("startup")
def startup():
    global VECTOR_STORE
    try:
        build_or_load_index()
    except Exception as e:
        print(f"Index build failed at startup: {e}")
        VECTOR_STORE = []


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks_indexed": len(VECTOR_STORE), "api_key_configured": bool(groq_client)}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on the server.")

    top_chunks = retrieve(req.message)
    context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in top_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context below. If the answer isn't in the context, say you "
        "don't have that information rather than guessing. Be concise.\n\n"
        "CONTEXT:\n" + (context if context else "No relevant context found.")
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(req.history[-6:])  # keep a little conversational memory
    messages.append({"role": "user", "content": req.message})

    completion = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = completion.choices[0].message.content
    sources = sorted(set(c["source"] for c in top_chunks))
    return {"answer": answer, "sources": sources}


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
