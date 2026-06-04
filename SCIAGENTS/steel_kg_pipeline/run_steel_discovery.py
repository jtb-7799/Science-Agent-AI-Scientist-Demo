#!/usr/bin/env python3
"""
Standalone SciAgents discovery for ultra-high strength steel.
Loads graph directly (bypasses SciAgents package import issues).
Runs the full pipeline: Path 鈫?Ontologist 鈫?Scientist 鈫?7 Expansions 鈫?Critic.
"""
import os, sys, json, re, time, pickle, csv
from datetime import datetime
from collections import defaultdict

# --- 1. Load Steel Graph Directly ---
print("=" * 60)
print("SciAgents Discovery - Ultra-High Strength Steel")
print("=" * 60)

print("\n[1/6] Loading steel graph...")
import networkx as nx
G = nx.read_graphml("/root/autodl-tmp/steel_kg_pipeline/graph_output/ultra_high_strength_steel.graphml")
G = nx.Graph(G)
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load embeddings
with open("/root/autodl-tmp/steel_kg_pipeline/graph_output/embeddings_ultra_high_strength_steel.pkl", "rb") as f:
    node_embeddings = pickle.load(f)
print(f"  {len(node_embeddings)} embeddings")

# Load BGE model for keyword matching
print("\n[2/6] Loading BGE model for keyword matching...")
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
bge_model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bge_model = bge_model.to(device)
print("  BGE model loaded")

# --- Helper: Find best matching node ---
def find_best_node(keyword, top_k=3):
    inputs = tokenizer(keyword, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bge_model(**inputs)
        kw_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    scores = []
    for node, emb in node_embeddings.items():
        sim = float(np.dot(kw_emb, emb) / (max(np.linalg.norm(kw_emb) * np.linalg.norm(emb), 1e-10)))
        scores.append((node, sim))
    return sorted(scores, key=lambda x: -x[1])[:top_k]

# --- Helper: Find path with randomization ---
import random
def sample_path(G, kw1, kw2, num_waypoints=2):
    """Find a path between two keywords with random waypoints."""
    nodes1 = find_best_node(kw1)
    nodes2 = find_best_node(kw2)
    n1 = nodes1[0][0]
    n2 = nodes2[0][0]
    print(f"  Best match: '{kw1}' 鈫?'{n1}'")
    print(f"  Best match: '{kw2}' 鈫?'{n2}'")

    # Add random waypoints
    all_nodes = list(G.nodes())
    waypoints = [n1]
    for _ in range(num_waypoints):
        wp = random.choice(all_nodes)
        waypoints.append(wp)
    waypoints.append(n2)

    # Build path segment by segment
    full_path = []
    for i in range(len(waypoints) - 1):
        try:
            seg = nx.shortest_path(G, waypoints[i], waypoints[i+1])
            if i == 0:
                full_path.extend(seg)
            else:
                full_path.extend(seg[1:])
        except nx.NetworkXNoPath:
            continue

    # Format as string
    parts = []
    for i in range(len(full_path) - 1):
        edge_data = G.get_edge_data(full_path[i], full_path[i+1])
        rel = str(edge_data.get("title", "related_to")) if edge_data else "related_to"
        rel = rel.split(";")[0].strip()
        parts.append(f"{full_path[i]} -- {rel} -- {full_path[i+1]}")
    return full_path, " -- ".join(parts)

# --- 3. Generate Path ---
print("\n[3/6] Sampling knowledge path...")
path_nodes, path_string = sample_path(G, "martensitic transformation", "precipitation strengthening")
print(f"  Path ({len(path_nodes)} nodes): {path_string[:200]}...")

# --- 4. Azure GPT-4o ---
print("\n[4/6] Setting up Azure GPT-4o...")
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="your-azure-openai-api-key",
    api_version="2024-12-01-preview",
    azure_endpoint="https://your-resource-name.openai.azure.com/",
)

def call_gpt(sys_prompt, user_prompt, max_tok=2048, temp=0.2):
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temp, max_tokens=max_tok,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            if attempt < 3: time.sleep(5)
    return ""

# --- 5. Run the Pipeline ---
print("\n[5/6] Running full SciAgents pipeline...")

# 5a. Ontologist
print("  [Ontologist] Defining concepts...")
onto_prompt = f"""You are a sophisticated ontologist specializing in ultra-high strength steel.

Given key concepts from a knowledge graph, define each term and discuss the relationships.

Knowledge graph path:
{path_string}

First define each term. Then discuss each relationship with scientific context from steel metallurgy.

Format:
### Definitions
(define each term)

### Relationships
(discuss each relationship)"""

onto = call_gpt("You are a steel metallurgy ontologist. Provide accurate, detailed responses.", onto_prompt, max_tok=1024, temp=0.1)
print(f"    {len(onto)} chars")

# 5b. Scientist - 7-aspect proposal
print("  [Scientist] Crafting research proposal...")
sci_prompt = f"""You are a sophisticated scientist in ultra-high strength steel research.

Given the following ontology analysis of a knowledge graph, synthesize a novel 7-aspect research proposal. Be quantitative with numbers, formulas, processing conditions.

{onto}

Output ONLY valid JSON:
{{"hypothesis": "...", "outcome": "...", "mechanisms": "...", "design_principles": "...", "unexpected_properties": "...", "comparison": "...", "novelty": "..."}}"""

sci = call_gpt("You are a creative steel scientist. Output valid JSON only.", sci_prompt, max_tok=2048, temp=0.2)
match = re.search(r"\{.*\}", sci, re.DOTALL) if sci else None
proposal = json.loads(match.group(0)) if match else {"raw": sci or "failed"}
print(f"    {len(proposal)} aspects")

# 5c. Expand each aspect
print("  [Expansion] Expanding 7 aspects...")
expanded = {}
aspects = [k for k in proposal.keys() if k != "raw"]
for i, field in enumerate(aspects):
    print(f"    {i+1}/7: {field}...")
    expand_prompt = f"""Carefully expand on the '{field}' aspect of this ultra-high strength steel research proposal.

Critically assess and improve the original. Add specifics, quantitative info, rationale, step-by-step reasoning. Comment on modeling/simulation techniques and experimental methods.

Original:
{proposal.get(field, '')}

Begin with: ### Expanded {field}"""

    result = call_gpt("You are a critical materials scientist. Provide detailed, quantitative responses.", expand_prompt, max_tok=1024, temp=0.2)
    if result:
        expanded[field] = result

# 5d. Critic
print("  [Critic] Reviewing proposal...")
complete = f"Proposal:\n{json.dumps(proposal, ensure_ascii=False, indent=2)}\n\nExpanded:\n{json.dumps(expanded, ensure_ascii=False, indent=2)}"
critic_prompt = f"""Read this ultra-high strength steel research proposal:

{complete[:8000]}

Provide:
(1) Summary (one paragraph with mechanisms, technologies, methods)
(2) Critical review with strengths, weaknesses, improvements
(3) Most impactful molecular modeling/simulation question with detailed steps
(4) Most impactful experimental question with detailed steps"""

critic = call_gpt("You are a critical materials science reviewer. Be thorough.", critic_prompt, max_tok=2048, temp=0.1)
print(f"    {len(critic)} chars")

# --- 6. Save ---
print("\n[6/6] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

md = f"""# SciAgents Discovery - Ultra-High Strength Steel

**Date:** {timestamp}
**Keywords:** martensitic transformation 鈫?precipitation strengthening
**Graph:** {G.number_of_nodes()} nodes, {G.number_of_edges()} edges

---

## 1. Knowledge Graph Path

{path_string}

---

## 2. Ontologist Analysis

{onto}

---

## 3. Research Proposal

{json.dumps(proposal, ensure_ascii=False, indent=2)}

---

## 4. Expanded Aspects

"""
for field, content in expanded.items():
    md += f"### {field}\n\n{content}\n\n---\n\n"

md += f"""## 5. Critical Review

{critic}
"""

data_dir = "/root/autodl-tmp/steel_kg_pipeline/results"
os.makedirs(data_dir, exist_ok=True)
md_path = f"{data_dir}/steel_discovery_{timestamp}.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\n{'='*60}")
print(f"Saved: {md_path}")
print(f"Total: {len(md)} chars")
print(f"{'='*60}")
print("\n=== PREVIEW ===\n")
print(md[:4000])

