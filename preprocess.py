import csv
import os
import random
from typing import Final, List, Optional

import networkx as nx
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from config import KNN_NEIGHBORS, STRUCTURAL_NEIGHBORS

M_STRUCTURAL: Final[int] = STRUCTURAL_NEIGHBORS
M_KNN: Final[int] = KNN_NEIGHBORS


def _patch_pyg_pickle_compat():
    try:
        import torch_geometric.data.data as pyg_data
    except ImportError:
        return

    try:
        from torch_geometric.data.storage import EdgeAttr, TensorAttr
    except Exception:
        EdgeAttr = object
        TensorAttr = object

    if not hasattr(pyg_data, "DataEdgeAttr"):
        class DataEdgeAttr(EdgeAttr):
            pass
        pyg_data.DataEdgeAttr = DataEdgeAttr

    if not hasattr(pyg_data, "DataTensorAttr"):
        class DataTensorAttr(TensorAttr):
            pass
        pyg_data.DataTensorAttr = DataTensorAttr


def _torch_load(path, **kwargs):
    _patch_pyg_pickle_compat()
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _load_node_ids(root_path: str, dataset_name: str, num_nodes: int) -> Optional[List[str]]:
    node_info_path = os.path.join(root_path, dataset_name, "node_info.csv")
    if not os.path.exists(node_info_path):
        print(f"Warning: node_info.csv not found at {node_info_path}; falling back to data.raw_texts.")
        return None

    with open(node_info_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "paper_id" not in reader.fieldnames:
            print(f"Error: {node_info_path} missing paper_id column.")
            return None
        node_ids = [str(row["paper_id"]).strip() for row in reader]

    if len(node_ids) != num_nodes:
        print(f"Error: node_info paper_id count ({len(node_ids)}) does not match num_nodes ({num_nodes}).")
        return None

    if len(set(node_ids)) != len(node_ids):
        print(f"Error: duplicate paper_id values found in {node_info_path}.")
        return None

    return node_ids


class PromptConfig:
    def __init__(
        self,
        ROOT_PATH,
        DATASET_NAME,
        thought,
        use_structural_prompt: bool,
        epoch: int,
    ):
        self.use_structural_prompt = use_structural_prompt
        self.ROOT_PATH = ROOT_PATH
        self.DATASET_NAME = DATASET_NAME
        self.thought = thought
        self.epoch = epoch


def load_pyg_data_for_prompt(dataset_name: str, root_path: str):
    pt_path = os.path.join(root_path, dataset_name, "processed_data.pt")

    if not os.path.exists(pt_path):
        print(f"Error: File not found: {pt_path}")
        return None, None, None

    try:
        data_list = _torch_load(pt_path)
        data = data_list[0] if isinstance(data_list, list) else data_list

        if not hasattr(data, "y") or not hasattr(data, "x"):
            print("Error: Data object missing attributes.")
            return None, None, None

        all_node_ids = _load_node_ids(root_path, dataset_name, data.num_nodes)
        if all_node_ids is None:
            if not hasattr(data, "raw_texts"):
                print("Error: no node_info paper_id list and data.raw_texts is missing.")
                return None, None, None
            all_node_ids = [str(node_id) for node_id in data.raw_texts]

        labels = data.y.tolist()
        id2label = dict(zip(all_node_ids, [str(label) for label in labels]))

        print(f"Loaded PyG Data. Nodes: {data.num_nodes}, Features: {data.x.shape}")
        print(f"Node ID count: {len(all_node_ids)}")
        return data, id2label, all_node_ids

    except Exception as e:
        print(f"Error loading processed_data.pt: {e}")
        return None, None, None


def build_graph_from_pyg(data: Data, all_node_ids: List[str]) -> Optional[nx.Graph]:
    print("-> Building NetworkX graph from PyG edge_index...")
    try:
        edge_index = data.edge_index.cpu().numpy()

        src_nodes = [all_node_ids[i] for i in edge_index[0]]
        dst_nodes = [all_node_ids[i] for i in edge_index[1]]

        G = nx.Graph()
        G.add_nodes_from(all_node_ids)
        G.add_edges_from(zip(src_nodes, dst_nodes))

        print(f"   - Graph built. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        return G

    except Exception as e:
        print(f"Error building graph: {e}")
        return None


def load_embeddings(DATA_PATH: str, thought_num: int, epoch: int, root_path: str = "dataset"):
    filename = f"{epoch}/{thought_num}_thought_embeddings.pt"
    full_path = os.path.join(root_path, DATA_PATH, filename)

    if not os.path.exists(full_path):
        print(f"Error: Embeddings file not found: {full_path}")
        return None

    try:
        embed = torch.load(full_path).detach().cpu().numpy().astype(np.float32)
        print(f"Embeddings loaded: {full_path}, Shape: {embed.shape}")
        return embed
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


def find_structural_neighbors(G: nx.Graph, start_node: str):
    if start_node not in G:
        return [], []

    first_hop = list(G.neighbors(start_node))
    second_hop = set()

    for node in first_hop:
        for connected_node in G.neighbors(node):
            if connected_node != start_node and connected_node not in first_hop:
                second_hop.add(connected_node)

    return first_hop, list(second_hop)


def get_knn_neighbors(target_node, features_matrix, all_node_ids, k_neighbors):
    id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
    if target_node not in id_to_idx:
        return []

    target_idx = id_to_idx[target_node]
    target_feature_vector = features_matrix[target_idx].reshape(1, -1)

    knn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
    knn_model.fit(features_matrix)

    indices = knn_model.kneighbors(target_feature_vector, n_neighbors=k_neighbors + 1, return_distance=False)
    return [all_node_ids[idx] for idx in indices.flatten()[1:]]


def process_node_and_generate_prompt(target_node, G, id2label, fusion_feature, all_node_ids, config):
    result = {
        "target_node": target_node,
        "output_text": id2label.get(target_node, "Category not found"),
    }

    if result["output_text"] == "Category not found":
        return None

    structural_set = set()
    result["prompt_structural"] = ""
    result["structural_neighbors"] = []

    if config.use_structural_prompt:
        hop_1_all, hop_2_all = find_structural_neighbors(G, target_node)
        selected_hop_1 = random.sample(hop_1_all, min(len(hop_1_all), M_STRUCTURAL))
        selected_hop_2 = random.sample(hop_2_all, min(len(hop_2_all), M_STRUCTURAL))
        structural_neighbors = selected_hop_1 + selected_hop_2
        structural_set = set(structural_neighbors)

        structural_str = [f"<{node}>" for node in structural_neighbors]
        result["prompt_structural"] = (
            f"Central node: <{target_node}>. "
            f"Selected structural neighbors (1-hop and 2-hop): [{', '.join(structural_str)}]."
        )
        result["structural_neighbors"] = structural_neighbors

    all_knn_f = get_knn_neighbors(target_node, fusion_feature, all_node_ids, M_KNN)

    content_knn_neighbors_f = []
    for node in all_knn_f:
        if not config.use_structural_prompt or node not in structural_set:
            content_knn_neighbors_f.append(node)
            if len(content_knn_neighbors_f) >= M_KNN:
                break

    fusion_knn_str = [f"<{node}>" for node in content_knn_neighbors_f]
    result["prompt_fusion_knn"] = (
        f"Central node: <{target_node}>. "
        f"Selected content-based neighbors (from fusion features, non-structural): [{', '.join(fusion_knn_str)}]."
    )
    result["knn_fusion"] = content_knn_neighbors_f

    return result


def generate_prompts_dataset(config: PromptConfig):
    prompt_dir = os.path.join(config.ROOT_PATH, config.DATASET_NAME, "prompt")
    os.makedirs(prompt_dir, exist_ok=True)

    fusion_path = os.path.join(prompt_dir, f"{config.DATASET_NAME}_fusion_knn_prompts.csv")
    structural_path = os.path.join(prompt_dir, f"{config.DATASET_NAME}_structural_prompts.csv")
    thought_path = os.path.join(prompt_dir, f"{config.DATASET_NAME}_prompts_thought_{config.thought}.csv")

    pyg_data, id2label, all_node_ids = load_pyg_data_for_prompt(config.DATASET_NAME, config.ROOT_PATH)
    if pyg_data is None:
        return None

    G = build_graph_from_pyg(pyg_data, all_node_ids)
    if G is None:
        return None

    emb = load_embeddings(config.DATASET_NAME, config.thought, config.epoch, config.ROOT_PATH)
    if emb is None:
        print("Error: Fusion features not loaded.")
        return None

    if config.thought == 1:
        prompt_jobs = [("fusion", fusion_path, False)]
        if config.use_structural_prompt:
            prompt_jobs.append(("structural", structural_path, True))

        for flag, path, use_structural in prompt_jobs:
            if os.path.exists(path):
                print(f"File exists, skipping: {path}")
                continue
            print(f"-> Generating {flag} prompts: {path}")
            write_prompts_csv(all_node_ids, G, id2label, emb, config, path, use_structural)
        return fusion_path

    print(f"-> Generating Thought {config.thought} fusion prompts: {thought_path}")
    if os.path.exists(thought_path):
        print(f"File exists, skipping: {thought_path}")
        return thought_path

    write_prompts_csv(all_node_ids, G, id2label, emb, config, thought_path, False)
    return thought_path


def write_prompts_csv(all_node_ids, G, id2label, emb0, config, output_path, use_structural):
    processed_count = 0
    skipped_not_in_graph = 0
    fieldnames = ["paper_id", "output_text", "prompt_text"]

    temp_config = PromptConfig(config.ROOT_PATH, config.DATASET_NAME, config.thought, use_structural, config.epoch)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for node_id in all_node_ids:
            if node_id not in G:
                skipped_not_in_graph += 1
                continue

            result = process_node_and_generate_prompt(node_id, G, id2label, emb0, all_node_ids, temp_config)
            if result is None:
                continue

            prompt_text = result["prompt_structural"] if use_structural else result["prompt_fusion_knn"]
            writer.writerow({
                "paper_id": node_id,
                "output_text": id2label.get(node_id, ""),
                "prompt_text": prompt_text,
            })

            processed_count += 1
            if processed_count % 500 == 0:
                print(f"  -> Processed {processed_count} / {len(all_node_ids)}")

    print(f"Written to {output_path}, Total: {processed_count}, Skipped: {skipped_not_in_graph}")
