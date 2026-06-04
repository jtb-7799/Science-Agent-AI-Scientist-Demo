from ScienceDiscovery.utils import *
import os

data_dir_source='./graph_giant_component/'

# --- Override for ultra-high strength steel graph ---
STEEL_GRAPH_DIR = '/root/autodl-tmp/steel_kg_pipeline/graph_output'
if os.path.exists(f'{STEEL_GRAPH_DIR}/ultra_high_strength_steel.graphml'):
    data_dir_source = STEEL_GRAPH_DIR + '/'
    embeddings_name = 'embeddings_ultra_high_strength_steel.pkl'
    graph_name = 'ultra_high_strength_steel.graphml'
# --- end override ---

tokenizer_model="BAAI/bge-large-en-v1.5"

embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
embedding_model = AutoModel.from_pretrained(tokenizer_model)

# Load graph (try direct load first, fall back to text-as-JSON for compatibility)
_graph_path = f'{data_dir_source}{graph_name}'
import networkx as nx
try:
    G = nx.read_graphml(_graph_path)
    G = nx.Graph(G)
    print(f"Graph loaded directly: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
except Exception:
    G = load_graph_with_text_as_JSON(data_dir=data_dir_source, graph_name=graph_name)
G = return_giant_component_of_graph(G)
G = nx.Graph(G)

# Load embeddings
try:
    import pickle
    with open(f'{data_dir_source}{embeddings_name}', 'rb') as f:
        node_embeddings = pickle.load(f)
    print(f"Embeddings loaded: {len(node_embeddings)}")
except Exception:
    try:
        node_embeddings = load_embeddings(f'{data_dir_source}{embeddings_name}')
    except:
        print("Node embeddings not loaded, regenerating...")
        node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model)