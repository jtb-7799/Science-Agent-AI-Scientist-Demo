"""
Simple test: verify steel graph + path finding works without importing SciAgents.
"""
import pickle
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import numpy as np

print("Loading steel graph...")
G = nx.read_graphml("steel_kg_pipeline/graph_output/ultra_high_strength_steel.graphml")
G = nx.Graph(G)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("Loading embeddings...")
with open("steel_kg_pipeline/graph_output/embeddings_ultra_high_strength_steel.pkl", "rb") as f:
    node_embeddings = pickle.load(f)
print(f"Embeddings: {len(node_embeddings)} nodes")

print("Loading BGE model...")
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Manual keyword → best node matching via embeddings
def find_best_node(keyword, embeddings, tokenizer, model, top_k=3):
    device = next(model.parameters()).device
    inputs = tokenizer(keyword, padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        kw_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    scores = []
    for node, emb in embeddings.items():
        sim = 1 - float(np.dot(kw_emb, emb) / (np.linalg.norm(kw_emb) * np.linalg.norm(emb)))
        scores.append((node, sim))
    return sorted(scores, key=lambda x: x[1])[:top_k]

# Simple shortest path
import torch

print("\n=== Testing keyword-to-node matching ===")
test_keywords = [
    "martensitic transformation",
    "precipitation strengthening",
    "grain refinement",
    "retained austenite",
    "hydrogen embrittlement",
    "quenching and partitioning",
    "mechanical properties",
    "yield strength",
    "ductility",
    "dislocation density",
]

for kw in test_keywords:
    matches = find_best_node(kw, node_embeddings, tokenizer, model)
    top_match = matches[0]
    print(f"  '{kw}' → '{top_match[0]}' (score: {top_match[1]:.3f})")

print("\n=== Testing path finding ===")
test_pairs = [
    ("martensitic transformation", "precipitation strengthening"),
    ("grain refinement", "ductility"),
    ("retained austenite", "TRIP effect"),
]

for kw1, kw2 in test_pairs:
    print(f"\nPath: '{kw1}' → '{kw2}'")
    matches1 = find_best_node(kw1, node_embeddings, tokenizer, model)
    matches2 = find_best_node(kw2, node_embeddings, tokenizer, model)

    n1 = matches1[0][0]
    n2 = matches2[0][0]
    print(f"  Best nodes: '{n1}' → '{n2}'")

    try:
        path = nx.shortest_path(G, source=n1, target=n2)
        print(f"  Path length: {len(path)-1} edges")
        for i, node in enumerate(path):
            if i < len(path) - 1:
                edge_data = G.get_edge_data(node, path[i+1])
                rel = edge_data.get("title", "??") if edge_data else "??"
                print(f"    {node[:50]} --[{rel[:40]}]--> ", end="")
            else:
                print(f"{node[:50]}")
    except nx.NetworkXNoPath:
        print(f"  No path found between these nodes!")
    except Exception as e:
        print(f"  Error: {e}")

print("\n=== Graph Stats ===")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
degrees = sorted([d for _, d in G.degree()], reverse=True)[:20]
print(f"Top 20 degree nodes: {[(n, d) for n, d in sorted(G.degree(), key=lambda x: -x[1])[:20]]}")
