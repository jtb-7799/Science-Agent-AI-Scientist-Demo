"""
Step 3: Build the ultra-high strength steel knowledge graph from extracted triples
and generate BGE node embeddings. Output files ready for SciAgents consumption.
"""
import json
import os
import pickle
import csv
import numpy as np
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import time


def load_triples(triples_dir: str = "steel_kg_pipeline/triples"):
    """Load all extracted triples from JSONL or batch CSV files."""
    all_triples = []

    # Prefer JSONL format
    jsonl_path = os.path.join(triples_dir, "all_triples.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line.strip())
                    if "node_1" in t and "node_2" in t and "edge" in t:
                        all_triples.append(t)
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(all_triples)} triples from JSONL")
        return all_triples

    # Fallback: CSV format
    csv_path = os.path.join(triples_dir, "triples_final.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_triples.append(row)
        print(f"Loaded {len(all_triples)} triples from CSV")
        return all_triples

    print("No triples found! Run 02_extract_triples.py first.")
    return []


def build_graph(triples: list) -> nx.Graph:
    """Build a NetworkX graph from triples, with deduplication and edge weighting."""
    # Aggregate edges
    edge_data = defaultdict(lambda: {"chunk_ids": set(), "edges": set(), "count": 0})
    node_texts = defaultdict(set)

    for t in triples:
        n1 = str(t["node_1"]).strip().lower()
        n2 = str(t["node_2"]).strip().lower()
        edge = str(t.get("edge", "")).strip()
        chunk_id = str(t.get("chunk_id", ""))

        if not n1 or not n2:
            continue

        key = tuple(sorted([n1, n2]))
        edge_data[key]["chunk_ids"].add(chunk_id)
        edge_data[key]["edges"].add(edge)
        edge_data[key]["count"] += 1
        if chunk_id:
            node_texts[n1].add(chunk_id)
            node_texts[n2].add(chunk_id)

    # Build graph
    G = nx.Graph()
    for (n1, n2), data in edge_data.items():
        edge_label = "; ".join(sorted(data["edges"])[:3])  # Top 3 labels
        weight = min(data["count"] / 2, 10.0)  # Capped weight
        chunk_ids = ",".join(sorted(data["chunk_ids"]))
        G.add_edge(n1, n2, title=edge_label, weight=weight, chunk_id=chunk_ids)

    # Add node texts
    for node, cids in node_texts.items():
        G.nodes[node]["texts"] = list(cids)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def generate_embeddings(G: nx.Graph, tokenizer, model, batch_size: int = 32):
    """Generate BGE embeddings for all graph nodes."""
    nodes = list(G.nodes())
    print(f"Generating embeddings for {len(nodes)} nodes...")

    embeddings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token pooling
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for node, emb in zip(batch, batch_embeddings):
            embeddings[node] = emb

        if (i + batch_size) % 500 == 0:
            print(f"  {min(i + batch_size, len(nodes))}/{len(nodes)} nodes embedded")

    print(f"Generated {len(embeddings)} embeddings")
    return embeddings


def clean_graph(G: nx.Graph, min_degree: int = 2) -> nx.Graph:
    """Remove isolated nodes and tiny components."""
    # Remove nodes with degree < min_degree
    low_deg = [n for n, d in G.degree() if d < min_degree]
    G.remove_nodes_from(low_deg)
    print(f"Removed {len(low_deg)} low-degree nodes")

    # Keep giant component
    if nx.is_connected(G):
        return G
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not components:
        return G
    giant = components[0]
    removed_nodes = [n for n in G.nodes() if n not in giant]
    G_giant = G.subgraph(giant).copy()
    print(f"Removed {len(removed_nodes)} nodes outside giant component "
          f"(small components: {len(components) - 1})")
    print(f"Giant component: {G_giant.number_of_nodes()} nodes, "
          f"{G_giant.number_of_edges()} edges")
    return G_giant


def main():
    print("=== Step 3: Building Ultra-High Strength Steel Knowledge Graph ===\n")

    # 1. Load triples
    triples = load_triples()
    if not triples:
        print("No triples found! Run 02_extract_triples.py first.")
        return

    # 2. Build graph
    print("\n--- Building graph ---")
    G = build_graph(triples)

    # 3. Clean graph
    print("\n--- Cleaning graph ---")
    G = clean_graph(G, min_degree=2)

    # 4. Generate embeddings
    print("\n--- Generating embeddings ---")
    tokenizer_model = "BAAI/bge-large-en-v1.5"
    print(f"Loading tokenizer and model: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    model = AutoModel.from_pretrained(tokenizer_model)

    embeddings = generate_embeddings(G, tokenizer, model)

    # 5. Save
    output_dir = "steel_kg_pipeline/graph_output"
    os.makedirs(output_dir, exist_ok=True)

    # Save graph as GraphML (with text as JSON string attributes)
    graph_path = f"{output_dir}/ultra_high_strength_steel.graphml"
    # Copy graph to avoid modifying original
    G_out = G.copy()
    for node, attrs in G_out.nodes(data=True):
        # Convert lists/dicts to JSON strings for GraphML compatibility
        for key, val in list(attrs.items()):
            if isinstance(val, (list, dict, set)):
                G_out.nodes[node][key] = json.dumps(val, ensure_ascii=False)
    nx.write_graphml(G_out, graph_path)
    print(f"Graph saved to {graph_path}")

    # Save embeddings
    emb_path = f"{output_dir}/embeddings_ultra_high_strength_steel.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {emb_path}")

    # Print stats
    print(f"\n=== Final Graph Statistics ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        print(f"Average degree: {np.mean(degrees):.2f}")
        print(f"Max degree: {max(degrees)}")
        print(f"Density: {nx.density(G):.6f}")

        # Top 10 nodes by degree
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 nodes by degree:")
        for node, deg in top_nodes:
            print(f"  {node}: {deg}")

    return G, embeddings


if __name__ == "__main__":
    G, embeddings = main()
