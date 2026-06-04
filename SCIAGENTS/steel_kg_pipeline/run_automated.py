#!/usr/bin/env python3
"""
Automated SciAgents discovery on ultra-high strength steel graph.
Uses AutoGen group chat with 13 agents.
"""
import os, sys, time, json, re, markdown2, pdfkit
from datetime import datetime

os.environ['OPENAI_API_KEY'] = 'sk-placeholder'
os.environ['SEMANTIC_SCHOLAR_API_KEY'] = 'your-semantic-scholar-api-key'

sys.path.insert(0, '/root/autodl-tmp/SciAgentsDiscovery')

print("Loading SciAgents + AutoGen + Steel Graph...", flush=True)
print("(BGE model loading on CPU, takes ~5 min)", flush=True)
t0 = time.time()

import ScienceDiscovery.graph as sg
import ScienceDiscovery.agents as am

print(f"Loaded in {time.time()-t0:.0f}s", flush=True)
print(f"Graph: {sg.G.number_of_nodes()} nodes, {sg.G.number_of_edges()} edges", flush=True)
print(f"Agents: {len(am.groupchat.agents)}", flush=True)
for a in am.groupchat.agents:
    print(f"  - {a.name}", flush=True)

print("\n" + "=" * 60)
print("Starting AutoGen Group Chat (max 50 rounds)")
print("Topic: Ultra-High Strength Steel Research Discovery")
print("=" * 60 + "\n")

# Reset agent states
am.planner.reset()
am.assistant.reset()
am.ontologist.reset()
am.scientist.reset()
am.critic_agent.reset()

data_dir = "/root/autodl-tmp/steel_kg_pipeline/results"
os.makedirs(data_dir, exist_ok=True)

# Run the group chat
try:
    res = am.user.initiate_chat(
        recipient=am.manager,
        message='Develop a novel research proposal for ultra-high strength steel materials. Use random concepts from the knowledge graph to generate a path, then analyze it and create a full proposal. After all aspects are expanded and reviewed, rate the novelty and feasibility.',
        clear_history=True,
    )
except Exception as e:
    print(f"Group chat error: {e}")
    import traceback
    traceback.print_exc()

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Collect messages from group chat
formatted = ""
for msg in am.groupchat.messages:
    content = msg.get('content', '')
    role = msg.get('name', 'unknown')
    if content:
        formatted += f"### {role}\n\n{content}\n\n"

md_path = f"{data_dir}/steel_automated_{timestamp}.md"
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(f"# SciAgents Automated Discovery - Ultra-High Strength Steel\n\n")
    f.write(f"**Date:** {timestamp}\n")
    f.write(f"**Graph:** {sg.G.number_of_nodes()} nodes\n")
    f.write(f"**Agents:** {len(am.groupchat.agents)}\n\n---\n\n")
    f.write(formatted)

print(f"\nSaved: {md_path} ({len(formatted)} chars)")

# PDF
try:
    html = markdown2.markdown(formatted)
    pdf_path = f"{data_dir}/steel_automated_{timestamp}.pdf"
    pdfkit.from_string(f"<style>body{{font-size:10px}}</style>{html}", pdf_path)
    print(f"PDF: {pdf_path}")
except Exception as e:
    print(f"PDF skipped: {e}")

print("DONE!")

