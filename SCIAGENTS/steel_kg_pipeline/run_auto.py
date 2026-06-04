#!/usr/bin/env python3
"""
Run SciAgents automated multi-agent discovery with steel graph.
Approach: use __init__.py's star import, then access what we need.
"""
import os, sys, json, re, markdown2, pdfkit
from datetime import datetime

os.environ['OPENAI_API_KEY'] = 'sk-placeholder'
os.environ['SEMANTIC_SCHOLAR_API_KEY'] = 'your-semantic-scholar-api-key'

sys.path.insert(0, '/root/autodl-tmp/SciAgentsDiscovery')

# Use star import like the original notebook
from ScienceDiscovery import *

# Access agents via the module's namespace
# (agents module is at ScienceDiscovery/agents.py, GraphReasoning also has agents.py)
# After 'from ScienceDiscovery import *', names are directly available
print("Checking available agents...")
print(f"  planner: {type(planner).__name__ if 'planner' in dir() else 'NOT FOUND'}")
print(f"  user: {type(user).__name__ if 'user' in dir() else 'NOT FOUND'}")
print(f"  manager: {type(manager).__name__ if 'manager' in dir() else 'NOT FOUND'}")

# The groupchat is also imported
print(f"  groupchat: {type(groupchat).__name__}")

print(f"\nAgents in groupchat:")
for a in groupchat.agents:
    print(f"  - {a.name}")

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Reset agents
planner.reset()
assistant.reset()
ontologist.reset()
scientist.reset()
critic_agent.reset()

# Output dir
data_dir_output = '/root/autodl-tmp/steel_kg_pipeline/results/'

# Run the group chat
print("\n" + "=" * 60)
print("Starting AutoGen Group Chat with Steel Graph...")
print("=" * 60 + "\n")

res = user.initiate_chat(
    recipient=manager,
    message='Develop a novel research proposal for ultra-high strength steel materials using random concepts from the knowledge graph. After the proposal is complete, rate the novelty and feasibility.',
    clear_history=True
)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

formatted_text = ""
for i, msg in enumerate(groupchat.messages):
    content = msg.get('content', '')
    role = msg.get('name', 'unknown')
    if content:
        formatted_text += f"### {role}\n\n{content}\n\n"

md_path = f"{data_dir_output}/steel_auto_discovery_{timestamp}.md"
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(f"# SciAgents Automated Discovery - Ultra-High Strength Steel\n\n")
    f.write(f"**Date:** {timestamp}\n")
    f.write(f"**Agents:** {len(groupchat.agents)}\n\n---\n\n")
    f.write(formatted_text)

print(f"\nSaved: {md_path}")

# PDF
try:
    html = markdown2.markdown(formatted_text)
    pdf_path = f"{data_dir_output}/steel_auto_discovery_{timestamp}.pdf"
    pdfkit.from_string(f"<style>body{{font-size:10px}}</style>{html}", pdf_path)
    print(f"PDF: {pdf_path}")
except Exception as e:
    print(f"PDF skipped: {e}")

