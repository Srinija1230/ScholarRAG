# 📓 Research Assistant

A RAG-powered research paper assistant with semantic chunking, RAGAS evaluation, UMAP visualisation, and support for multiple LLM providers.

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
streamlit run app.py
```

---

## 🤖 Supported Models

| Provider | Model | Cost |
|---|---|---|
| Google | Gemini 2.5 Flash Preview | Free |
| Anthropic | Claude Haiku 4.5 | Free tier |
| Meta via Groq | Llama 3.3 70B Versatile | Free |
| OpenAI | GPT-4o | Paid |

---

## 🔧 Updating Model Versions

All model strings are in one place at the top of `config.py`:

```python
# ── UPDATE MODEL VERSIONS HERE ──────────────────────────────────
# Google:    https://ai.google.dev/gemini-api/docs/models
# Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/overview
# Groq:      https://console.groq.com/docs/models
# OpenAI:    https://platform.openai.com/docs/models

GEMINI_CHAT_MODEL  = "gemini-2.5-flash-preview-03-25"
CLAUDE_CHAT_MODEL  = "claude-haiku-4-5-20251001"
LLAMA_CHAT_MODEL   = "llama-3.3-70b-versatile"
OPENAI_CHAT_MODEL  = "gpt-4o"
```

To update any model — change that one line, nothing else.

---

## ✨ Features

- **Semantic Chunking** — splits paper text by meaning, not fixed word count
- **RAGAS Evaluation** — faithfulness, answer relevancy, context precision scores
- **UMAP Visualisation** — explore the embedding space of all indexed chunks
- **Multi-Model** — switch between Gemini, Claude, Llama, GPT-4o from the sidebar
- **Caching** — pipeline results cached per topic so rebuilds are optional

---

## ⚠️ Known Constraints

### arXiv HTML availability
- Full paper text is fetched via arXiv's HTML endpoint (`arxiv.org/html/{id}`)
- Available for most papers published **after ~2018**
- Papers before ~2018 automatically fall back to **abstract only** — answers will be less detailed for those
- A small number of recent papers may also lack HTML versions (malformed submissions etc.) — same abstract fallback applies

### arXiv rate limiting
- arXiv asks for polite API usage — a 0.5s delay is added between paper fetches
- Fetching 15 papers takes ~8–10 seconds on first run (cached after that)
- Do not remove the delay or arXiv may temporarily block your IP

### Embedding fallback
- **Claude and Llama (Groq)** do not have embedding APIs
- When using these providers, embeddings automatically fall back to **Google Gemini**
- This means a valid `GOOGLE_API_KEY` is required regardless of which chat model you pick

### Paper search
- Papers are sourced from **arXiv only** — covers CS, Physics, Math, Biology, Economics
- Does not search PubMed, IEEE, ACM, or other databases
- Citation counts are not available from arXiv (shown as 0)
