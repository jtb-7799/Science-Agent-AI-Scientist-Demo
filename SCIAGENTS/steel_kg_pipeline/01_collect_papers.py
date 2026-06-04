"""
Step 1: Collect ~1000 ultra-high strength steel papers from Semantic Scholar API.
Uses multiple keyword queries to get broad coverage of the field.
"""
import requests
import json
import time
import os
import pandas as pd
from datetime import datetime

# Search queries covering key ultra-high strength steel subfields
QUERIES = [
    "ultra-high strength steel alloy design",
    "martensitic steel transformation strengthening",
    "precipitation hardening steel nanoparticles",
    "grain refinement ultra-fine steel",
    "TRIP steel transformation induced plasticity",
    "TWIP steel twinning induced plasticity",
    "medium manganese steel retained austenite",
    "maraging steel precipitation intermetallic",
    "HSLA steel microalloyed carbide nitride",
    "dual-phase steel ferrite martensite",
    "quenching partitioning steel carbon",
    "bainitic steel austempering carbide-free",
    "high strength low alloy steel processing",
    "nanostructured bainitic steel",
    "steel dislocation density strengthening mechanism",
    "hydrogen embrittlement high strength steel",
    "fatigue behavior ultra-high strength steel",
    "severe plastic deformation nanocrystalline steel",
    "steel composition optimization machine learning",
    "steel mechanical properties microstructure relation",
]

# Additional broader queries to reach ~1000 papers
BROAD_QUERIES = [
    "advanced high strength steel automotive",
    "steel heat treatment microstructure evolution",
    "steel phase transformation kinetics",
    "steel alloying element effect mechanical properties",
    "steel thermomechanical processing",
]

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
API_KEY = "your-semantic-scholar-api-key"
HEADERS = {"x-api-key": API_KEY}


def search_papers(query: str, limit: int = 50, offset: int = 0) -> list:
    """Search Semantic Scholar for papers matching a query. Rate: 1 req/s."""
    time.sleep(1.05)  # Respect 1 request/second rate limit
    params = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "fields": "title,abstract,year,authors,externalIds,publicationTypes,url",
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
        elif resp.status_code == 429:
            print(f"  Rate limited on query '{query}', waiting 5s...")
            time.sleep(5)
            return search_papers(query, limit, offset)
        else:
            print(f"  Error {resp.status_code} for query '{query}': {resp.text[:100]}")
            return []
    except Exception as e:
        print(f"  Exception for query '{query}': {e}")
        time.sleep(2)
        return []


def collect_papers():
    all_papers = {}
    total = 0

    print(f"=== Collecting ultra-high strength steel papers ===")
    print(f"Specific queries: {len(QUERIES)}, Broad queries: {len(BROAD_QUERIES)}")

    for query in QUERIES + BROAD_QUERIES:
        print(f"\nSearching: '{query}'")
        for offset in [0, 50]:
            papers = search_papers(query, limit=50, offset=offset)
            new_count = 0
            for p in papers:
                paper_id = p.get("paperId")
                if paper_id and paper_id not in all_papers:
                    if p.get("abstract"):  # Only keep papers with abstracts
                        authors = p.get("authors") or []
                        all_papers[paper_id] = {
                            "paperId": str(paper_id),
                            "title": str(p.get("title") or ""),
                            "abstract": str(p.get("abstract") or ""),
                            "year": int(p.get("year")) if p.get("year") else None,
                            "authors": ", ".join([a.get("name", "") for a in authors]),
                            "url": str(p.get("url") or ""),
                            "query": str(query),
                        }
                        new_count += 1
            print(f"  Found {new_count} new papers (total: {len(all_papers)})")
            if len(all_papers) >= 1000:
                break
        if len(all_papers) >= 1000:
            break

    print(f"\n=== Collected {len(all_papers)} unique papers with abstracts ===")

    papers_list = list(all_papers.values())
    os.makedirs("steel_kg_pipeline", exist_ok=True)

    # Save as JSON (primary storage)
    json_path = "steel_kg_pipeline/steel_papers.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers_list, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {json_path}")

    # Save as CSV manually to avoid pandas version issues
    csv_path = "steel_kg_pipeline/steel_papers.csv"
    import csv as csv_module
    fields = ["paperId", "title", "abstract", "year", "authors", "url", "query"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in papers_list:
            writer.writerow({k: p.get(k, "") for k in fields})
    print(f"Saved CSV to {csv_path}")

    # Year distribution
    years = [p["year"] for p in papers_list if p.get("year")]
    from collections import Counter
    year_counts = Counter(years)
    print("\nYear distribution:")
    for year, count in sorted(year_counts.items(), reverse=True)[:20]:
        print(f"  {year}: {count}")

    return papers_list


if __name__ == "__main__":
    collect_papers()

