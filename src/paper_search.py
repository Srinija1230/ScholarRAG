import time, requests
import xml.etree.ElementTree as ET
import pandas as pd

def _search_arxiv(query: str, max_results: int = 30) -> list:
    """
    Fetch papers from arXiv API for a single query.
    Returns up to max_results papers sorted by relevance.
    """
    try:
        response = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results":  max_results,
                "sortBy":       "relevance",
            },
            timeout=15
        )
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(response.text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            year     = int(entry.find("atom:published", ns).text[:4])
            papers.append({
                "title":     entry.find("atom:title", ns).text.strip().replace("\n", " "),
                "authors":   ", ".join(a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)[:4]),
                "abstract":  entry.find("atom:summary", ns).text.strip().replace("\n", " "),
                "year":      year,
                "citations": 0,
                "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
                "arxiv_id":  arxiv_id,
                "source":    "arxiv",
            })
        return papers
    except:
        return []

def search_all_sources(search_queries: list) -> pd.DataFrame:
    """
    Search arXiv across all query variants, deduplicate, return DataFrame.
    Fetches 30 results per query (increased from 20 for better coverage).
    """
    all_papers = []
    for query in search_queries:
        print(f"  Searching arXiv: {query}")
        all_papers.extend(_search_arxiv(query, max_results=30))
        time.sleep(0.4)   # polite delay for arXiv API

    papers_df = pd.DataFrame(all_papers)
    if papers_df.empty:
        return papers_df

    papers_df = papers_df.drop_duplicates("arxiv_id").reset_index(drop=True)
    print(f"  → {len(papers_df)} unique papers found")
    return papers_df
