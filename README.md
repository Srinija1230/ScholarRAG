# 📚 ScholarRAG

A RAG-powered research paper assistant with semantic chunking, LLM-based RAGAS evaluation, UMAP visualisation, and support for multiple LLM providers. Served via a Flask web app.

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your API keys

Open `config.py` and fill in your keys:

```python
GOOGLE_API_KEY    = "your_key_here"   # Free  → https://aistudio.google.com
ANTHROPIC_API_KEY = "your_key_here"   # Free tier → https://console.anthropic.com
GROQ_API_KEY      = "your_key_here"   # Free  → https://console.groq.com
OPENAI_API_KEY    = "your_key_here"   # Paid  → https://platform.openai.com
```

> **Note:** At minimum you need a **Google API key** — it is used for embeddings by all providers (Anthropic and Groq do not have embedding APIs).

### 3. Run

```bash
python web_app.py
```

The app runs on `http://localhost:5000` by default.

---

## 🤖 Supported Models

| Provider | Model | Cost |
|---|---|---|
| Google | Gemini 2.5 Flash | Free |
| Anthropic | Claude Haiku 4.5 | Free tier |
| Meta via Groq | Llama 3.3 70B Versatile | Free |
| OpenAI | GPT-4o | Paid |

Switch between models from the sidebar dropdown at any time — no restart needed.

---

## 🔧 Updating Model Versions

All model strings live in one place at the top of `config.py`:

```python
# ── UPDATE MODEL VERSIONS HERE ──────────────────────────────────────────────
# Google:    https://ai.google.dev/gemini-api/docs/models
# Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/overview
# Groq:      https://console.groq.com/docs/models
# OpenAI:    https://platform.openai.com/docs/models

GEMINI_CHAT_MODEL  = "gemini-2.5-flash"
CLAUDE_CHAT_MODEL  = "claude-haiku-4-5-20251001"
LLAMA_CHAT_MODEL   = "llama-3.3-70b-versatile"
OPENAI_CHAT_MODEL  = "gpt-4o"
```

To update any model — change that one line, nothing else.

---

## ✨ Features

- **Multi-Model** — switch between Gemini, Claude, Llama, and GPT-4o from the sidebar
- **Semantic Chunking** — splits paper text by meaning, not fixed word count
- **Diverse Retrieval** — samples the best chunks from every indexed paper so follow-up questions aren't limited to previously surfaced papers
- **Cross-Paper Reranking** — retrieved chunks are reranked by relevance before being passed to the LLM
- **LLM-based RAGAS Evaluation** — optional faithfulness, answer relevancy, and context precision scores (toggle in sidebar)
- **UMAP Visualisation** — explore the embedding space of all indexed chunks
- **Conversation History** — past sessions are listed in the sidebar and can be reloaded
- **Caching** — pipeline results are cached per topic so rebuilds are optional

---

## 🗂️ Project Structure

```
├── web_app.py          # Flask server — routes and HTML template
├── pipeline.py         # Orchestrates the ingestion and QA pipeline
├── config.py           # API keys, model strings, pipeline settings
├── requirements.txt
├── src/
│   ├── topic_expander.py   # Expands user topic into multiple search queries
│   ├── paper_search.py     # Searches arXiv and returns paper metadata
│   ├── pdf_parser.py       # Fetches and parses paper text (HTML or abstract)
│   ├── chunker.py          # Semantic + fixed-size chunking
│   ├── embedder.py         # Wraps embedding API calls
│   ├── vector_store.py     # FAISS vector store with diverse search
│   ├── reranker.py         # LLM-based chunk reranking
│   ├── answer_generator.py # Prompt templates and answer generation
│   ├── evaluator.py        # LLM-based RAGAS scoring
│   └── visualizer.py       # UMAP dimensionality reduction + Plotly chart
└── data/
    └── cache/              # Per-topic FAISS index, metadata, and raw JSON
```

---

## ⚙️ Pipeline Settings

Tunable in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `MAX_PAPERS_TO_SELECT` | 15 | Final papers kept after ranking |
| `INITIAL_RETRIEVAL_COUNT` | 45 | Wide retrieval pool to ensure diversity |
| `FINAL_RERANKED_COUNT` | 10 | Chunks passed to the LLM after reranking |
| `WORDS_PER_CHUNK` | 600 | Target words per semantic chunk |
| `CHUNK_WORD_OVERLAP` | 80 | Overlap between fixed-size chunks (fallback) |
| `TOP_CHUNKS_PER_PAPER` | 2 | Chunks sampled per paper in diverse retrieval |

---

## ⚠️ Known Constraints

### arXiv HTML availability
- Full paper text is fetched via arXiv's HTML endpoint (`arxiv.org/html/{id}`)
- Available for most papers published **after ~2018**
- Papers before ~2018 automatically fall back to **abstract only** — answers will be less detailed for those
- A small number of recent papers may also lack HTML versions (malformed submissions etc.) — same abstract fallback applies

### arXiv rate limiting
- arXiv requests polite API usage — a 0.5 s delay is added between paper fetches
- Fetching 15 papers takes ~8–10 seconds on first run (cached after that)
- Removing the delay risks a temporary IP block from arXiv

### Embedding fallback
- **Claude and Llama (Groq)** do not provide embedding APIs
- When using these providers, embeddings automatically fall back to **Google Gemini**
- A valid `GOOGLE_API_KEY` is therefore required regardless of which chat model is selected

### Paper sources
- Papers are sourced from **arXiv only** — covers CS, Physics, Math, Biology, and Economics
- PubMed, IEEE, ACM, and other databases are not searched
- Citation counts are not available from arXiv (shown as 0)

---

## 🖼️ Screenshot
<img width="1470" height="956" alt="ScholarRAG UI" src="https://github.com/user-attachments/assets/3b2d3d79-e98b-45eb-b08b-355806f5431d" />

