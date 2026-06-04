#!/bin/bash
set -e
cd /root/autodl-tmp

echo "=========================================="
echo "  Ultra-High Strength Steel SciAgents Pipeline"
echo "=========================================="

# Step 2: Extract triples (if not already done)
if [ ! -f "steel_kg_pipeline/triples/triples_final.csv" ]; then
    echo -e "\n>>> Step 2: Extracting triples with Azure GPT-4o..."
    python3 steel_kg_pipeline/02_extract_triples.py
else
    echo -e "\n>>> Step 2: SKIPPED (triples already extracted)"
fi

# Step 3: Build graph + embeddings
echo -e "\n>>> Step 3: Building knowledge graph..."
python3 steel_kg_pipeline/03_build_graph.py

# Step 4: Quick test
echo -e "\n>>> Step 4: Testing pipeline..."
python3 steel_kg_pipeline/04_test_pipeline.py

echo -e "\n=========================================="
echo "  Pipeline Complete!"
echo "  Run discovery: python3 steel_kg_pipeline/run_discovery.py --kw1 'martensitic steel' --kw2 'high toughness'"
echo "=========================================="
