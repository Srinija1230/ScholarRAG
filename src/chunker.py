import re
import numpy as np
import config
from config import WORDS_PER_CHUNK, CHUNK_WORD_OVERLAP

def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def _split_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip().split()) >= 5]

def _semantic_chunks(sentences: list, embeddings: list, threshold: float = 0.75) -> list:
    if not sentences:
        return []
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        current.append(sentences[i])
        if sim < threshold or len(" ".join(current).split()) >= WORDS_PER_CHUNK:
            chunks.append(" ".join(current))
            current = [sentences[i]]
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 30]

def _fixed_chunks(text: str) -> list:
    """Fallback: fixed word-size chunks with overlap."""
    words  = text.split()
    chunks = []
    for start in range(0, len(words), WORDS_PER_CHUNK - CHUNK_WORD_OVERLAP):
        chunk = " ".join(words[start:start + WORDS_PER_CHUNK])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
    return chunks

def _embed_sentences(sentences: list) -> list:
    """Always use Gemini for sentence embeddings (Claude/Llama have no embed API)."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GOOGLE_API_KEY)
        result = genai.embed_content(
            model=config.GEMINI_EMBED_MODEL,
            content=sentences,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        print(f"  [chunker] embedding failed, using fixed chunks: {e}")
        return None

def chunk_single_paper(paper: dict) -> list:
    paper_sections = paper.get("sections") or {"body": paper.get("full_text", "")}
    paper_chunks   = []
    chunk_index    = 0

    for section_name, section_text in paper_sections.items():
        if not section_text or len(section_text.split()) < 30:
            continue

        sentences  = _split_sentences(section_text)
        method     = "fixed"
        chunks     = []

        if sentences:
            embeddings = _embed_sentences(sentences)
            if embeddings:
                chunks = _semantic_chunks(sentences, embeddings)
                method = "semantic"

        if not chunks:
            chunks = _fixed_chunks(section_text)

        for chunk_text in chunks:
            paper_chunks.append({
                "chunk_id":     f"{paper['title'][:20].replace(' ','_')}_chunk{chunk_index:03d}",
                "text":         chunk_text,
                "paper_title":  paper["title"],
                "authors":      paper.get("authors", ""),
                "year":         paper.get("year"),
                "section":      section_name,
                
                "paper_url":    paper.get("paper_url", ""),
                "chunk_method": method,
            })
            chunk_index += 1

    return paper_chunks

def chunk_all_papers(ingested_papers: list) -> list:
    """Chunk all papers and return in memory — nothing saved to disk."""
    all_chunks = []
    for paper in ingested_papers:
        paper_chunks = chunk_single_paper(paper)
        all_chunks.extend(paper_chunks)
        method = paper_chunks[0].get("chunk_method", "?") if paper_chunks else "none"
        print(f"  {paper['title'][:50]} → {len(paper_chunks)} chunks [{method}]")
    print(f"  → {len(all_chunks)} total chunks ready")
    return all_chunks
