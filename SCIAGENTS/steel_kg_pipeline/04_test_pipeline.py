#!/usr/bin/env python3
"""
Step 4: Run SciAgents multi-agent system on the ultra-high strength steel knowledge graph.
This tests the full pipeline end-to-end with a few example keyword pairs.
"""
import sys
import os

# Add SciAgents to path
sys.path.insert(0, "/root/autodl-tmp/SciAgentsDiscovery")

# Override graph paths BEFORE importing graph module
STEEL_GRAPH_DIR = "/root/autodl-tmp/steel_kg_pipeline/graph_output"
STEEL_GRAPH_NAME = "ultra_high_strength_steel.graphml"
STEEL_EMBEDDINGS_NAME = "embeddings_ultra_high_strength_steel.pkl"

print("=== SciAgents + Ultra-High Strength Steel Graph ===\n")
print(f"Graph: {STEEL_GRAPH_DIR}/{STEEL_GRAPH_NAME}")
print(f"Embeddings: {STEEL_GRAPH_DIR}/{STEEL_EMBEDDINGS_NAME}")

# Patch ScienceDiscovery.graph module to use our steel graph
import ScienceDiscovery.graph as sgraph
sgraph.data_dir_source = STEEL_GRAPH_DIR + "/"
sgraph.embeddings_name = STEEL_EMBEDDINGS_NAME
sgraph.graph_name = STEEL_GRAPH_NAME

# Reload graph with steel data
from transformers import AutoTokenizer, AutoModel
from ScienceDiscovery.utils import *

print("\n--- Loading Steel Knowledge Graph ---")
tokenizer = AutoTokenizer.from_pretrained(sgraph.tokenizer_model)
model = AutoModel.from_pretrained(sgraph.tokenizer_model)

G = load_graph_with_text_as_JSON(data_dir=STEEL_GRAPH_DIR + "/", graph_name=STEEL_GRAPH_NAME)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

G = return_giant_component_of_graph(G)
G = nx.Graph(G)
print(f"Giant component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load embeddings
try:
    node_embeddings = load_embeddings(f"{STEEL_GRAPH_DIR}/{STEEL_EMBEDDINGS_NAME}")
    print(f"Loaded {len(node_embeddings)} node embeddings")
except Exception as e:
    print(f"Embeddings error: {e}")
    print("Regenerating embeddings...")
    node_embeddings = generate_node_embeddings(G, tokenizer, model)

# Update graph module variables
sgraph.G = G
sgraph.node_embeddings = node_embeddings
sgraph.embedding_tokenizer = tokenizer
sgraph.embedding_model = model

# Test: find paths between steel-relevant keyword pairs
print("\n=== Testing Path Sampling ===\n")

test_pairs = [
    ("martensitic transformation", "precipitation strengthening"),
    ("grain refinement", "ductility"),
    ("retained austenite", "TRIP effect"),
    ("carbon content", "yield strength"),
    ("quenching and partitioning", "mechanical properties"),
]

for kw1, kw2 in test_pairs:
    print(f"\n--- Path: {kw1} -> {kw2} ---")
    try:
        path_vis, path_string = create_path(
            G, tokenizer, model, node_embeddings,
            keyword_1=kw1, keyword_2=kw2,
            randomness_factor=0.2, num_random_waypoints=2,
            shortest_path=False, verbatim=False, data_dir="./"
        )
        print(f"  Path: {path_string[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")

print("\n=== Pipeline Ready ===")
print("""
To run the full multi-agent research hypothesis generation, use:
    python3 steel_kg_pipeline/04_run_discovery.py --kw1 "martensite" --kw2 "toughness"

This will:
  1. Sample a path between the given keywords in the steel knowledge graph
  2. Run Ontologist -> Scientist -> 7 Expansion Agents -> Critic -> Novelty Check
  3. Output a complete research proposal as PDF/MD/CSV
""")
