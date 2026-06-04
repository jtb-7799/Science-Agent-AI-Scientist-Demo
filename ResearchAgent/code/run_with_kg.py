"""Runner using full knowledge graph from 15-paper paper pool."""
import sys, json
from tqdm import tqdm

from utils import data_io
from knowledge.store import KnowledgeStore
from models.anthropic import AnthropicClient
from pipelines.research_pipeline import ResearchPipeline


MANUAL_ABSTRACTS = {
    255115362: (
        "A 2.5 GPa grade maraging steel strengthened by ultrahigh density of interweaved "
        "shearable nanostructures is reported. The nanostructures consist of fully coherent "
        "B2-Ni(Al, Fe) nanoprecipitates and Mo-rich disordered clusters with approximately "
        "30 at.% Mo content. Both nanostructures possess coherent interfaces with the "
        "martensitic matrix and can be sheared by dislocations during deformation. Atom probe "
        "tomography reveals the co-precipitation mechanism. The composite nanostructure enables "
        "the ultra-strong steel to maintain plasticity exceeding 6%."
    ),
    287560955: (
        "A strategy to make ultra-strong as-quenched carbon martensitic steel ductile. Severe "
        "anisotropic lattice distortion with high tetragonality (c/a ratio up to 1.033) is "
        "deliberately introduced using high concentrations of substitutional solutes (Ni, Co, Al) "
        "and carbon (0.38 wt%) in a Fe-15.7Co-22Ni-2.96Al-0.38C alloy. This ultra-high "
        "tetragonality reduces twin boundary energy and activates deformation twinning as an "
        "additional plastic carrier, overcoming the inherent brittleness of as-quenched martensite. "
        "The alloy achieves yield strength 1.64 GPa, ultimate tensile strength 2.42 GPa, "
        "uniform elongation 6.46%."
    ),
}


def run(paper_ids, knowledge_store, api_client, output_path):
    # Fetch bare paper info via S2 batch API
    import requests, os
    headers = {"x-api-key": os.environ.get("S2_API_KEY", "")}
    r = requests.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        headers=headers,
        params={"fields": "corpusId,paperId,title,abstract,year"},
        json={"ids": paper_ids},
        timeout=30
    )
    raw_papers = r.json() if r.status_code == 200 else []

    papers = []
    for p in raw_papers:
        if not isinstance(p, dict) or not p.get('title'):
            continue
        cid = p.get('corpusId')
        if not p.get('abstract') and cid in MANUAL_ABSTRACTS:
            p['abstract'] = MANUAL_ABSTRACTS[cid]
        if p.get('title') and p.get('abstract'):
            papers.append(p)

    print(f"Papers ready: {len(papers)}/{len(raw_papers)}")
    results = []

    for paper in tqdm(papers):
        context = {'paper': {key: paper.get(key) for key in ('title', 'abstract')}}

        # Get knowledge graph entities for this paper via co-occurrence
        entities = knowledge_store.get_relevant_entities([paper['corpusId']])
        print(f"  Knowledge entities: {len(entities)}")

        # No S2 references (all elided), but knowledge graph provides context
        context.update(references=[], entities=entities)

        research_pipeline = ResearchPipeline(api_client=api_client)
        context = research_pipeline.run(context)

        results.append(context)
        data_io.save_result(output_path, context)

    return results


if __name__ == "__main__":
    import argparse, os
    os.environ.setdefault('S2_API_KEY', os.environ.get('S2_API_KEY', ''))

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data-path', '-d', default='./data/maraging/papers.jsonl')
    argparser.add_argument('--knowledge-path', '-k', default='./data/maraging/knowledge.jsonl')
    argparser.add_argument('--output-path', '-o', default='./results/maraging_ideas.jsonl')
    argparser.add_argument('--model-name', '-m', default='deepseek-chat')
    args = argparser.parse_args()

    paper_ids = data_io.load_paper_ids(args.data_path, num_papers=10)
    print(f"Paper IDs: {paper_ids}")

    knowledge_store = KnowledgeStore(args.knowledge_path)
    print(f"Knowledge store: {len(knowledge_store.paper2entities)} papers, "
          f"{len(knowledge_store.entity_counter)} entities, "
          f"{sum(len(v) for v in knowledge_store.entity_cooccurrence.values())} co-occurrence pairs")

    api_client = AnthropicClient(model=args.model_name)
    results = run(paper_ids, knowledge_store, api_client, args.output_path)
