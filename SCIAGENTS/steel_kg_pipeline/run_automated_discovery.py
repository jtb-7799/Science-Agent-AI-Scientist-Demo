#!/usr/bin/env python3
"""
Run the full automated SciAgentsDiscovery pipeline on ultra-high strength steel graph.
Uses direct module attribute access to avoid name conflicts.
"""
import os, sys, time, json, re, markdown2, pdfkit
from datetime import datetime

# --- API Keys ---
os.environ['OPENAI_API_KEY'] = 'sk-placeholder'
os.environ['SEMANTIC_SCHOLAR_API_KEY'] = 'your-semantic-scholar-api-key'

# --- Output dir ---
data_dir_output = '/root/autodl-tmp/steel_kg_pipeline/results/'
os.makedirs(data_dir_output, exist_ok=True)

# --- Import SciAgents ---
sys.path.insert(0, '/root/autodl-tmp/SciAgentsDiscovery')

# Import graph first (already patched for steel)
import ScienceDiscovery.graph as sg
print(f"Graph: {sg.G.number_of_nodes()} nodes, {sg.G.number_of_edges()} edges")
try:
    print(f"Embeddings: {len(sg.node_embeddings)} loaded")
except:
    print("Embeddings not loaded yet")

# Import agents module (not star import to avoid name clashes)
import ScienceDiscovery.agents as agents_mod
import ScienceDiscovery.utils as utils_mod

# --- Prepare AutoGen group chat ---
print("\n=== Starting Automated SciAgents Multi-Agent Discovery ===")
print("Agents: Planner, Assistant, Ontologist, Scientist,")
print("        7 expansion agents, Critic, Novelty Checker")
print("Graph: Ultra-High Strength Steel (2114 nodes, 4346 edges)")
print("=" * 60)

# Reset agent states
agents_mod.planner.reset()
agents_mod.assistant.reset()
agents_mod.ontologist.reset()
agents_mod.scientist.reset()
agents_mod.critic_agent.reset()

# Run the group chat
print("\nInitiating group chat...\n")
res = agents_mod.user.initiate_chat(
    recipient=agents_mod.manager,
    message='Develop a novel research proposal for ultra-high strength steel materials using random concepts from the knowledge graph. After the proposal is complete, rate the novelty and feasibility.',
    clear_history=True
)

# --- Save results ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Extract messages from groupchat history
formatted_text = ""
for i, msg in enumerate(agents_mod.groupchat.messages):
    content = msg.get('content', '')
    role = msg.get('name', 'unknown')
    if content:
        formatted_text += f"### {role}\n\n{content}\n\n"

md_path = f"{data_dir_output}/steel_automated_discovery_{timestamp}.md"
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(f"# SciAgents Automated Discovery - Ultra-High Strength Steel\n\n")
    f.write(f"**Date:** {timestamp}\n\n")
    f.write(f"**Graph:** {sg.G.number_of_nodes()} nodes, {sg.G.number_of_edges()} edges\n\n")
    f.write("---\n\n")
    f.write(formatted_text)

print(f"\n{'='*60}")
print(f"Results saved to: {md_path}")
print(f"{'='*60}")

# Try PDF
try:
    html_content = markdown2.markdown(formatted_text)
    css = "<style>body { font-size: 10px; }</style>"
    full_html = f"{css}{html_content}"
    pdf_path = f"{data_dir_output}/steel_automated_discovery_{timestamp}.pdf"
    pdfkit.from_string(full_html, pdf_path)
    print(f"PDF saved to: {pdf_path}")
except Exception as e:
    print(f"PDF skipped: {e}")

print("\nDone!")

