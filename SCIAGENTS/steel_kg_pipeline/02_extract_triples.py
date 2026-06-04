"""
Step 2: Extract knowledge triples from paper abstracts using Azure GPT-4o.
Optimized single-call per paper (~2s each = ~30 min total for 1000 papers).
Saves progress after every paper to allow resumption.
"""
import json
import time
import re
import os
import csv
from openai import AzureOpenAI
from typing import Optional

AZURE_API_KEY = "your-azure-openai-api-key"
AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4o"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

SYS_PROMPT = (
    "You are an expert in materials science and metallurgy. Extract key concepts "
    "and their relationships from the given paper title and abstract to build an "
    "ontology knowledge graph for ultra-high strength steel research.\n\n"
    "Output exactly a JSON array of 8-10 concept pairs with relationships:\n"
    '[\n'
    '  {"node_1": "concept_1", "node_2": "concept_2", "edge": "relationship"},\n'
    '  ...\n'
    ']\n\n'
    "Rules:\n"
    "- Use consistent, widely-recognized metallurgy terminology\n"
    "- node_1 and node_2 must be distinct concepts (materials, phases, mechanisms, properties, processes)\n"
    "- edge must be a concise verb phrase describing the relationship (e.g. 'strengthens', 'is phase of', 'measured by')\n"
    "- Include: alloy systems, phases, mechanical properties, processing methods, strengthening mechanisms, characterization techniques\n"
    "- Output ONLY the JSON array, no other text\n"
    "- Example format:\n"
    '[\n'
    '  {"node_1": "lath martensite", "node_2": "ultra-high strength", "edge": "provides"},\n'
    '  {"node_1": "carbon content", "node_2": "MS temperature", "edge": "lowers"},\n'
    '  {"node_1": "nanoprecipitates", "node_2": "dislocation motion", "edge": "impedes"}\n'
    ']\n'
)


def generate(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 3:
                time.sleep(5)
            else:
                print(f"\n  LLM error after retries: {e}")
    return ""


def extract_json(text: str) -> Optional[list]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main():
    with open("steel_kg_pipeline/steel_papers.json", "r") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers\n")

    output_dir = "steel_kg_pipeline/triples"
    os.makedirs(output_dir, exist_ok=True)

    progress_file = f"{output_dir}/progress.json"
    triples_file = f"{output_dir}/all_triples.jsonl"
    processed_ids = set()

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_ids = set(json.load(f))
        print(f"Resuming: {len(processed_ids)} already done, {len(papers) - len(processed_ids)} remaining\n")

    start_time = time.time()
    total_skipped = len(processed_ids)
    new_triples_count = 0

    for i, paper in enumerate(papers):
        pid = paper["paperId"]
        if pid in processed_ids:
            total_skipped -= 1
            continue

        text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
        if len(text) > 6000:
            text = text[:6000]  # Truncate very long abstracts

        prompt = f"Context: ```{text}```\n\nOutput the JSON array:"
        response = generate(prompt)
        triples = extract_json(response)

        if triples:
            with open(triples_file, "a", encoding="utf-8") as f:
                for t in triples:
                    t["chunk_id"] = pid
                    if "node_1" in t and "node_2" in t and "edge" in t:
                        f.write(json.dumps(t, ensure_ascii=False) + "\n")
                        new_triples_count += 1

        processed_ids.add(pid)
        print(".", end="", flush=True)

        # Save progress every 20 papers
        if (i + 1) % 20 == 0 or (i + 1) == len(papers):
            with open(progress_file, "w") as f:
                json.dump(list(processed_ids), f)
            elapsed = time.time() - start_time
            done = len(processed_ids)
            rate = (done - (len(processed_ids) - (i + 1 - total_skipped))) / max(elapsed, 1)
            remaining = len(papers) - done
            eta_min = remaining / max(rate, 0.01) / 60
            print(f" [{done}/{len(papers)}] ~{rate:.1f}/s ETA:{eta_min:.1f}min")

    # Convert JSONL to CSV for graph building
    all_triples = []
    if os.path.exists(triples_file):
        with open(triples_file, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line.strip())
                if "node_1" in t and "node_2" in t and "edge" in t:
                    all_triples.append(t)

    csv_path = f"{output_dir}/triples_final.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_1", "node_2", "edge", "chunk_id"])
        writer.writeheader()
        for t in all_triples:
            writer.writerow(t)
    print(f"\nDone. {len(all_triples)} triples saved to {csv_path}")


if __name__ == "__main__":
    main()

