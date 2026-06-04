#!/usr/bin/env python3
"""
Run the complete SciAgents discovery pipeline on the ultra-high strength steel
knowledge graph. Uses the same multi-agent system but with steel graph data.
"""
import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, "/root/autodl-tmp/SciAgentsDiscovery")

# --- Patch graph paths ---
STEEL_DIR = "/root/autodl-tmp/steel_kg_pipeline/graph_output"
import ScienceDiscovery.graph as sg
sg.data_dir_source = STEEL_DIR + "/"
sg.embeddings_name = "embeddings_ultra_high_strength_steel.pkl"
sg.graph_name = "ultra_high_strength_steel.graphml"

from transformers import AutoTokenizer, AutoModel
from ScienceDiscovery.utils import *
from ScienceDiscovery.agents import *
from ScienceDiscovery.llm_config import *
from ScienceDiscovery.graph import *

# --- Load steel graph ---
print("Loading ultra-high strength steel knowledge graph...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
model = AutoModel.from_pretrained(tokenizer_model)

G_local = load_graph_with_text_as_JSON(data_dir=STEEL_DIR + "/", graph_name=graph_name)
G_local = return_giant_component_of_graph(G_local)
G_local = nx.Graph(G_local)
print(f"Graph: {G_local.number_of_nodes()} nodes, {G_local.number_of_edges()} edges")

try:
    embeddings = load_embeddings(f"{STEEL_DIR}/{embeddings_name}")
    print(f"Loaded {len(embeddings)} embeddings")
except Exception as e:
    print(f"Regenerating embeddings: {e}")
    embeddings = generate_node_embeddings(G_local, tokenizer, model)

# --- Update agent functions to use steel graph ---
# We need to override the generate_path function registered in agents.py
# to use our steel graph data

def steel_generate_path(keyword_1=None, keyword_2=None) -> str:
    """Generate path using steel knowledge graph."""
    path_vis, path_string = create_path(
        G_local, tokenizer, model, embeddings,
        generate_graph_expansion=None,
        randomness_factor=0.2,
        num_random_waypoints=3,
        shortest_path=False,
        second_hop=False,
        data_dir="./",
        save_files=False,
        verbatim=True,
        keyword_1=keyword_1,
        keyword_2=keyword_2,
    )
    return path_string


# --- Re-register functions with steel graph ---
from autogen import register_function

# Clear old registrations and re-register
user._llm_config = False

# Patch the graph module globals
import ScienceDiscovery.graph as sgraph
sgraph.G = G_local
sgraph.node_embeddings = embeddings
sgraph.embedding_tokenizer = tokenizer
sgraph.embedding_model = model

# Re-register with updated globals
from ScienceDiscovery.graph import G, node_embeddings, embedding_tokenizer, embedding_model

def generate_path_steel(keyword_1=None, keyword_2=None) -> str:
    path_vis, path_string = create_path(
        G, embedding_tokenizer, embedding_model, node_embeddings,
        generate_graph_expansion=None,
        randomness_factor=0.2,
        num_random_waypoints=3,
        shortest_path=False,
        second_hop=False,
        data_dir="./",
        save_files=False,
        verbatim=True,
        keyword_1=keyword_1,
        keyword_2=keyword_2,
    )
    return path_string


def main():
    parser = argparse.ArgumentParser(description="SciAgents Steel Discovery")
    parser.add_argument("--kw1", type=str, default=None, help="First keyword")
    parser.add_argument("--kw2", type=str, default=None, help="Second keyword")
    parser.add_argument("--random", action="store_true", help="Use random keywords")
    parser.add_argument("--message", type=str, default=None, help="Research question for the group chat")
    args = parser.parse_args()

    if args.random:
        kw1, kw2 = None, None
        task_msg = "Generate a novel research proposal for ultra-high strength steel materials using the provided knowledge path."
    elif args.kw1 and args.kw2:
        kw1, kw2 = args.kw1, args.kw2
        task_msg = f"Generate a novel research proposal connecting '{kw1}' and '{kw2}' in ultra-high strength steel."
    else:
        kw1, kw2 = "martensitic transformation", "precipitation strengthening"
        task_msg = f"Generate a novel research proposal connecting '{kw1}' and '{kw2}' in the field of ultra-high strength steel materials."

    print(f"\nKeywords: {kw1 or 'random'} -> {kw2 or 'random'}")
    print(f"Task: {task_msg}\n")

    # First generate the path
    print("=== Sampling Knowledge Path ===")
    path_string = generate_path_steel(keyword_1=kw1, keyword_2=kw2)
    print(f"Path: {path_string[:300]}...")

    # Run the group chat with explicit task
    task_msg = f"""{task_msg}

Here is the knowledge path from the ultra-high strength steel materials graph:
{path_string}

Please analyze the nodes and relationships in this knowledge path, define each term, discuss the relationships, and then craft a detailed research proposal with all seven aspects (hypothesis, outcome, mechanisms, design_principles, unexpected_properties, comparison, novelty). Then expand each aspect, provide a critical review, and rate novelty/feasibility."""

    user.initiate_chat(
        manager,
        clear_history=True,
        message=task_msg,
        max_turns=50,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./steel_kg_pipeline/results"
    os.makedirs(output_dir, exist_ok=True)

    # Extract messages
    messages = []
    for msg in groupchat.messages:
        if msg.get("content"):
            messages.append({
                "role": msg.get("name", "unknown"),
                "content": msg.get("content", ""),
            })

    md_path = f"{output_dir}/steel_discovery_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# SciAgents Steel Discovery\n")
        f.write(f"Keywords: {kw1} -> {kw2}\n")
        f.write(f"Date: {timestamp}\n\n")
        f.write(f"## Knowledge Path\n{path_string}\n\n")
        for msg in messages:
            f.write(f"## {msg['role']}\n\n{msg['content']}\n\n")

    print(f"\nResults saved to {md_path}")


if __name__ == "__main__":
    main()
