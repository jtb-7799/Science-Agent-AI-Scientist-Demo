#!/usr/bin/env python3
"""
Run full SciAgents discovery pipeline on steel graph.
Uses the non-automated pipeline (research_generation from utils.py)
which bypasses AutoGen but produces the same 7-aspect proposal + expansion + critique.
"""
import os, sys, json, time, re
from datetime import datetime

os.environ['OPENAI_API_KEY'] = 'sk-placeholder'
os.environ['SEMANTIC_SCHOLAR_API_KEY'] = 'your-semantic-scholar-api-key'

sys.path.insert(0, '/root/autodl-tmp/SciAgentsDiscovery')

print("=" * 60)
print("SciAgents Full Discovery - Ultra-High Strength Steel")
print("=" * 60)

# --- Step 1: Load graph and embeddings without importing agents ---
print("\n[1/5] Loading steel knowledge graph...")
from ScienceDiscovery import graph as sg
# sg.graph.py already patched for steel graph
G = sg.G
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load/reload embeddings if needed
from ScienceDiscovery.utils import load_embeddings, generate_node_embeddings
try:
    node_embeddings = load_embeddings('/root/autodl-tmp/steel_kg_pipeline/graph_output/embeddings_ultra_high_strength_steel.pkl')
    print(f"  Embeddings: {len(node_embeddings)} loaded from file")
except:
    print("  Regenerating embeddings...")
    node_embeddings = generate_node_embeddings(G, sg.embedding_tokenizer, sg.embedding_model)
    print(f"  Generated {len(node_embeddings)} embeddings")

tokenizer = sg.embedding_tokenizer
model = sg.embedding_model

# --- Step 2: Define Azure generation function ---
print("\n[2/5] Initializing Azure GPT-4o...")

from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="your-azure-openai-api-key",
    api_version="2024-12-01-preview",
    azure_endpoint="https://your-resource-name.openai.azure.com/",
)

def generate_azure(system_prompt, prompt, max_tokens=2048, temperature=0.2):
    """Call Azure GPT-4o with retries."""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    Retry {attempt+1}: {e}")
            if attempt < 3:
                time.sleep(5)
    return ""

# --- Step 3: Generate path ---
print("\n[3/5] Sampling knowledge path...")

from ScienceDiscovery.utils import create_path

path_vis, path_string = create_path(
    G, tokenizer, model, node_embeddings,
    keyword_1="martensitic transformation",
    keyword_2="precipitation strengthening",
    randomness_factor=0.15, num_random_waypoints=2,
    shortest_path=False, verbatim=True, data_dir="./",
)
print(f"\n  Path: {path_string[:300]}...")

# --- Step 4: Generate research proposal ---
print("\n[4/5] Generating research proposal (Ontologist + Scientist)...")

# Ontologist
onto_prompt = f"""You are a sophisticated ontologist specializing in ultra-high strength steel metallurgy.

Given the following key concepts extracted from a knowledge graph, define each term and discuss the relationships identified in the graph.

Knowledge graph path (format: node_1 -- relationship -- node_2):

{path_string}

First, define each term in the knowledge graph. Then, discuss each relationship with scientific context from steel metallurgy and materials science.

Format:
### Definitions:
... clear definition of each term ...

### Relationships:
... thorough discussion of all relationships ..."""

print("  Calling Ontologist...")
onto_response = generate_azure(
    system_prompt="You are a metallurgy ontologist who provides accurate, detailed responses.",
    prompt=onto_prompt,
    max_tokens=1024, temperature=0.1,
)
if not onto_response:
    onto_response = "[Ontologist response failed]"
print(f"  Ontologist: {len(onto_response)} chars")

# Scientist
sci_prompt = f"""You are a sophisticated scientist trained in ultra-high strength steel research and innovation.

Given the definitions and relationships from a knowledge graph, synthesize a novel research proposal.

{onto_response}

Craft a detailed research proposal with 7 aspects. Be quantitative - include numbers, sequences, chemical formulas, processing conditions.

Output as JSON:
{{"hypothesis": "...", "outcome": "...", "mechanisms": "...", "design_principles": "...", "unexpected_properties": "...", "comparison": "...", "novelty": "..."}}"""

print("  Calling Scientist...")
sci_response = generate_azure(
    system_prompt="You are a creative steel scientist. Output valid JSON only.",
    prompt=sci_prompt,
    max_tokens=2048, temperature=0.2,
)

# Parse JSON proposal
match = re.search(r"\{.*\}", sci_response, re.DOTALL) if sci_response else None
proposal = {}
if match:
    try:
        proposal = json.loads(match.group(0))
        print(f"  Proposal: {len(proposal)} aspects generated")
    except:
        proposal = {"raw": sci_response}
else:
    proposal = {"raw": sci_response or "[Scientist response failed]"}

# --- Step 5: Expand each aspect + Critic ---
print("\n[5/5] Expanding aspects and reviewing...")

expanded = {}
for field_idx, field in enumerate(list(proposal.keys())[:7]):
    print(f"  Expanding: {field} ({field_idx+1}/7)...")

    expand_prompt = f"""Carefully expand on the '{field}' aspect of this research proposal for ultra-high strength steel.

Critically assess and improve the original content. Add more specifics, quantitative scientific information (chemical formulas, numbers, processing conditions, microstructures), rationale, and step-by-step reasoning.

Original {field}:
{proposal.get(field, '')}

Do not add introductory phrases. Begin with: ### Expanded {field}"""

    result = generate_azure(
        system_prompt="You are a critical materials scientist. Provide detailed, quantitative responses.",
        prompt=expand_prompt,
        max_tokens=1024, temperature=0.2,
    )
    if result:
        expanded[field] = result

# Critic
print("  Calling Critic...")
complete_text = f"""Research Proposal:
{json.dumps(proposal, ensure_ascii=False, indent=2)}

Expanded Aspects:
{json.dumps(expanded, ensure_ascii=False, indent=2)}"""

critic_prompt = f"""Read this research proposal for ultra-high strength steel:

{complete_text[:8000]}

Provide:
(1) Summary of the document (one paragraph with mechanisms, technologies, methods)
(2) Thorough critical scientific review with strengths, weaknesses, and improvements
(3) The single most impactful question for molecular modeling/simulation with detailed steps
(4) The single most impactful question for experimental/synthetic work with detailed steps"""

critic_response = generate_azure(
    system_prompt="You are a critical materials science reviewer. Be thorough and constructive.",
    prompt=critic_prompt,
    max_tokens=2048, temperature=0.1,
)

# --- Save ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

md_content = f"""# SciAgents Full Discovery - Ultra-High Strength Steel

**Date:** {timestamp}
**Keywords:** martensitic transformation 鈫?precipitation strengthening

---

## Knowledge Graph Path

{path_string}

---

## Ontologist Analysis

{onto_response}

---

## Research Proposal (7 Aspects)

{json.dumps(proposal, ensure_ascii=False, indent=2)}

---

## Expanded Aspects

"""
for field, content in expanded.items():
    md_content += f"### {field}\n\n{content}\n\n---\n\n"

md_content += f"""
## Critical Review & Recommendations

{critic_response}
"""

data_dir = "/root/autodl-tmp/steel_kg_pipeline/results"
os.makedirs(data_dir, exist_ok=True)
md_path = f"{data_dir}/steel_full_discovery_{timestamp}.md"

with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"\n{'='*60}")
print(f"Full discovery saved to: {md_path}")
print(f"Size: {len(md_content)} chars")
print(f"{'='*60}")

# Print a preview
print("\n=== PREVIEW (first 3000 chars) ===\n")
print(md_content[:3000])

