# Simple RAG Chatbot 
NOTE - This is a practise RAG CHATBOT where I learn to responsibly built a RAG CHATBOT, and very much for learning purpose.. Please feel free to use it for learning purposes. 

Also you can check on this link - https://rag-chatbot-practise.onrender.com (it only works for 7 days(for the api key to work) from 31-07-2026)

For the API , please replace your API key with the placeholder value. 

A minimal Retrieval-Augmented Generation chatbot: put text files in `docs/`,
it embeds them locally, and answers questions using only that content —
using a free chat API. No paid services involved.

## Stack
- **Embeddings:** run locally on the server via `fastembed` — a small
  open-source model, no API key, no cost, no usage limit.
- **Chat/generation:** [Groq](https://console.groq.com) — free tier, no
  credit card required, generous daily rate limits (model used here:
  `llama-3.1-8b-instant`). You just need a free account to get an API key.

## How it works
1. On startup, every `.txt`/`.md` file in `docs/` is split into chunks and
   embedded locally (first run downloads the small embedding model, ~130MB,
   automatically — after that it's cached).
2. Embeddings are also cached to `embeddings_cache.json` so restarts skip
   re-embedding unless you change the docs.
3. Each chat message is embedded and compared (cosine similarity) against
   all stored chunks; the top 4 most relevant chunks go into the prompt.
4. Groq's `llama-3.1-8b-instant` answers using that context and the
   response lists which source files it drew from.

## 1. Get a free Groq API key
1. Go to https://console.groq.com and sign up (no card needed).
2. Create an API key (starts with `gsk_...`).

## 2. Run locally

```bash
cd rag-chatbot
python -m venv venv
 source "venv\Scripts\activate"  # Bash: source venv/bin/activate     
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste in your real GROQ_API_KEY , this will rename the env example after you edit.

uvicorn main:app --reload
```

Open http://localhost:8000 — first run will take a bit longer since it
downloads the local embedding model once. Try asking about the content of
`docs/sample.txt`.

Replace/add files in `docs/` with your own content, then restart — it
re-embeds automatically since the cache is keyed on file modification time.

## 3. Deploy to Render

1. Push this folder to a GitHub repo (include your real `docs/*.txt` files;
   don't push your real `.env`, only `.env.example`).
2. In Render: **New + → Web Service** → connect your repo.
3. Render should auto-detect `render.yaml`. If not, set manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3
4. Under **Environment → Environment Variables**, add:
   - `GROQ_API_KEY` = your real key
5. Deploy. First deploy takes a little longer (downloading the embedding
   model + embedding your docs); subsequent restarts are faster.

## Notes / limitations (intentionally kept simple)
- Free Render web services spin down when idle — first request after
  inactivity will be slow (cold start, can be 30-60s). Normal, not a bug.
- Groq's free tier has daily/per-minute rate limits — plenty for a
  prototype/demo, not for production-scale traffic.
- Vector store is in-memory + a JSON cache file — fine for a handful of
  documents, not for thousands of pages.
- No authentication/rate-limiting on the app itself — add if this goes
  anywhere public-facing.
- To swap models later, only the `CHAT_MODEL` / `EMBED_MODEL_NAME`
  constants at the top of `main.py` need to change.
