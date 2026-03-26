# =============================================================================
# ── UPDATE MODEL VERSIONS HERE ───────────────────────────────────────────────
# Google:    https://ai.google.dev/gemini-api/docs/models
# Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/overview
# Groq:      https://console.groq.com/docs/models
# OpenAI:    https://platform.openai.com/docs/models
# =============================================================================
GEMINI_CHAT_MODEL    = "gemini-2.5-flash"
GEMINI_EMBED_MODEL   = "models/embedding-001"

CLAUDE_CHAT_MODEL    = "claude-haiku-4-5-20251001"

LLAMA_CHAT_MODEL     = "llama-3.3-70b-versatile"

OPENAI_CHAT_MODEL    = "gpt-4o"
OPENAI_EMBED_MODEL   = "text-embedding-3-small"
# =============================================================================

# --- Hardcoded API Keys ---
# Get free key → https://aistudio.google.com
GOOGLE_API_KEY    = "AIzaSyC3V4Uzpgsp9U1UYjy6x83cDKA57Fu1y8A"
# Get free key → https://console.anthropic.com
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"
# Get free key → https://console.groq.com
GROQ_API_KEY      = "YOUR_GROQ_API_KEY_HERE"
# Paid → https://platform.openai.com
OPENAI_API_KEY    = "YOUR_OPENAI_API_KEY_HERE"

# --- Active provider (overridden by UI dropdown) ---
ACTIVE_PROVIDER = "gemini"  # "gemini" | "claude" | "llama" | "openai"

# --- Model registry ---
MODEL_OPTIONS = {
    "gemini": {
        "label":       "Gemini 2.5 Flash — Google",
        "chat_model":  GEMINI_CHAT_MODEL,
        "embed_model": GEMINI_EMBED_MODEL,
        "api_key":     GOOGLE_API_KEY,
    },
    "claude": {
        "label":       "Claude Haiku 4.5 — Anthropic",
        "chat_model":  CLAUDE_CHAT_MODEL,
        "embed_model": None,   # No embedding API — falls back to Gemini
        "api_key":     ANTHROPIC_API_KEY,
    },
    "llama": {
        "label":       "Llama 3.3 70B — Meta via Groq",
        "chat_model":  LLAMA_CHAT_MODEL,
        "embed_model": None,   # No embedding API — falls back to Gemini
        "api_key":     GROQ_API_KEY,
    },
    "openai": {
        "label":       "GPT-4o — OpenAI",
        "chat_model":  OPENAI_CHAT_MODEL,
        "embed_model": OPENAI_EMBED_MODEL,
        "api_key":     OPENAI_API_KEY,
    },
}

def active_model():
    return MODEL_OPTIONS[ACTIVE_PROVIDER]

# =============================================================================
# ── PIPELINE SETTINGS ────────────────────────────────────────────────────────
# Tune these to balance speed vs answer quality.
# HTML fetching is fast so higher values are safe compared to PDF downloads.
# =============================================================================
MAX_PAPERS_TO_SELECT    = 15   # final papers kept after ranking (was 10)
INITIAL_RETRIEVAL_COUNT = 45   # wider pool ensures diversity across all 15 papers
FINAL_RERANKED_COUNT    = 10   # more chunks after reranking to cover more papers
WORDS_PER_CHUNK         = 600  # target words per semantic chunk
CHUNK_WORD_OVERLAP      = 80   # overlap between fixed-size chunks (fallback)
TOP_CHUNKS_PER_PAPER    = 2    # chunks sampled per paper in diverse retrieval
