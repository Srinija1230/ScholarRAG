import re, time, requests
import pandas as pd

ARXIV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://arxiv.org)"
}

SECTION_HEADING_PATTERN = re.compile(
    r"^(abstract|introduction|related work|background|method(?:ology)?|"
    r"experiment[s]?|result[s]?|evaluation|discussion|conclusion[s]?|limitation[s]?)$",
    re.IGNORECASE
)

def _split_into_sections(raw_text: str) -> dict:
    sections = {}
    current_section = "preamble"
    current_lines   = []
    for line in raw_text.split("\n"):
        if SECTION_HEADING_PATTERN.match(line.strip()):
            if current_lines:
                sections[current_section] = " ".join(current_lines).strip()
            current_section = line.strip().lower()
            current_lines   = []
        else:
            current_lines.append(line.strip())
    if current_lines:
        sections[current_section] = " ".join(current_lines).strip()
    return sections

def _fetch_html_text(arxiv_id: str) -> dict:
    """
    Fetch full paper text from arXiv HTML endpoint.
    Available for most papers published after ~2018.
    Falls back to abstract for older papers or if HTML is unavailable.
    """
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(html_url, headers=ARXIV_HEADERS, timeout=20)
        if resp.status_code != 200:
            return {"full_text": "", "sections": {}, "success": False}

        html = resp.text
        # Remove script, style, nav, header, footer tags and their content
        html = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL)
        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Decode common HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
                   .replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()

        if len(text.split()) < 200:
            return {"full_text": "", "sections": {}, "success": False}

        return {
            "full_text": text,
            "sections":  _split_into_sections(text),
            "success":   True,
            "source":    "html",
        }
    except Exception as e:
        print(f"    [html fetch] failed: {e}")
        return {"full_text": "", "sections": {}, "success": False}

def ingest_papers(papers_df: pd.DataFrame) -> list:
    """
    Fetch full text for each paper via arXiv HTML endpoint.
    Nothing is saved to disk — all text is kept in memory only.

    Constraints:
    - arXiv HTML available for most papers after ~2018; older ones fall back to abstract.
    - 0.5s delay between requests to respect arXiv rate limits.
    - Malformed HTML pages also fall back to abstract.
    """
    ingested = []

    for i, (_, row) in enumerate(papers_df.iterrows(), 1):
        title    = row["title"]
        arxiv_id = row.get("arxiv_id", "")
        abstract = row.get("abstract", "")

        print(f"  [{i}/{len(papers_df)}] {title[:65]}…")

        extracted = {"full_text": "", "sections": {}, "success": False}

        if arxiv_id:
            print(f"    → fetching HTML from arxiv.org/html/{arxiv_id}")
            extracted = _fetch_html_text(arxiv_id)
            if extracted["success"]:
                print(f"    ✅ HTML text fetched ({len(extracted['full_text'].split())} words)")
            else:
                print(f"    ⚠️  HTML unavailable (paper may be pre-2018) — using abstract")

        if not extracted["success"] and abstract:
            extracted = {
                "full_text":     abstract,
                "sections":      {"abstract": abstract},
                "success":       True,
                "source":        "abstract_fallback",
                "abstract_only": True,
            }

        ingested.append({
            "title":     title,
            "authors":   row.get("authors", ""),
            "year":      row.get("year"),
            "abstract":  abstract,
            "paper_url": row.get("paper_url", ""),
            **extracted,
        })
        time.sleep(0.5)  # polite delay — arXiv rate limits aggressive scrapers

    return ingested
