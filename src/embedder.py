import config

def _embed_with_gemini(texts: list, task: str = "retrieval_document") -> list:
    import google.generativeai as genai
    genai.configure(api_key=config.GOOGLE_API_KEY)
    all_emb = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        result = genai.embed_content(model=config.GEMINI_EMBED_MODEL, content=batch, task_type=task)
        all_emb.extend(result["embedding"])
    return all_emb

def _embed_with_openai(texts: list) -> list:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    all_emb = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        resp = client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=batch)
        all_emb.extend([d.embedding for d in resp.data])
    return all_emb

def embed_texts(texts: list, task: str = "retrieval_document") -> list:
    """Route embeddings — OpenAI uses its own, everything else uses Gemini."""
    try:
        if config.ACTIVE_PROVIDER == "openai":
            return _embed_with_openai(texts)
        else:
            return _embed_with_gemini(texts, task)
    except Exception as e:
        print(f"  [embed] failed: {e}")
        return [[0.0] * 768] * len(texts)
