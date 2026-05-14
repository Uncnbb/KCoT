import os

import numpy as np
import torch
from torch_geometric.data import Data

from config import DATASET_NAME, RANDOM_SEED, ROOT_PATH


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
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _load_raw_data(dataset_name, filename):
    _patch_pyg_pickle_compat()
    base_dir = os.path.join(ROOT_PATH, dataset_name)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = _torch_load(path)
    if not hasattr(data, "edge_index") and hasattr(data, "adj_t"):
        row, col, _ = data.adj_t.t().coo()
        data.edge_index = torch.stack([row, col], dim=0)
    return data, base_dir


def _load_simteg_features(base_dir, num_nodes):
    emb_files = ["simteg_sbert_x.pt", "simteg_roberta_x.pt", "simteg_e5_x.pt"]
    embs = []
    for filename in emb_files:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding missing: {path}")
        embs.append(_torch_load(path, map_location="cpu"))

    x = torch.cat(embs, dim=-1)
    if num_nodes is not None and x.shape[0] != num_nodes:
        raise ValueError(f"Feature row count {x.shape[0]} does not match node count {num_nodes}.")
    return x


def _edge_index_to_pairs(edge_index):
    pairs = set()
    if edge_index.shape[1] == 0:
        return pairs
    for src, dst in edge_index.t().tolist():
        if src == dst:
            continue
        u, v = (src, dst) if src < dst else (dst, src)
        pairs.add((u, v))
    return pairs


def _pairs_to_edge_index(pairs, undirected=False):
    edges = []
    for src, dst in sorted(pairs):
        edges.append((src, dst))
        if undirected:
            edges.append((dst, src))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _split_pairs(pairs, val_ratio=0.05, seed=RANDOM_SEED):
    pairs = list(sorted(pairs))
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    val_pairs = set(pairs[:val_count])
    train_pairs = set(pairs[val_count:])
    return train_pairs, val_pairs


def _sample_negative_pairs(num_nodes, positive_pairs, count, seed):
    rng = np.random.default_rng(seed)
    negatives = set()
    positive_pairs = set(positive_pairs)
    max_attempts = max(1000, count * 50)
    attempts = 0

    while len(negatives) < count and attempts < max_attempts:
        src = int(rng.integers(0, num_nodes))
        dst = int(rng.integers(0, num_nodes))
        attempts += 1
        if src == dst:
            continue
        u, v = (src, dst) if src < dst else (dst, src)
        pair = (u, v)
        if pair in positive_pairs or pair in negatives:
            continue
        negatives.add(pair)

    if len(negatives) < count:
        raise RuntimeError(f"Could only sample {len(negatives)} negative pairs out of {count}.")
    return negatives


def load_gnn_dataset(dataset_name=DATASET_NAME, task="nc"):
    if task == "nc":
        filename = "processed_data.pt"
    elif task == "lp":
        filename = "processed_data_link_notest.pt"
    else:
        raise ValueError(f"Unknown task: {task}")

    pyg_data, base_dir = _load_raw_data(dataset_name, filename)
    x_features = _load_simteg_features(base_dir, pyg_data.num_nodes)

    final_data = Data(
        x=x_features,
        edge_index=pyg_data.edge_index,
        y=getattr(pyg_data, "y", None),
        num_nodes=x_features.shape[0],
    )

    for key in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(pyg_data, key):
            setattr(final_data, key, getattr(pyg_data, key))
    return final_data


def load_lp_data_with_test_split(dataset_name=DATASET_NAME):
    full_data, _ = _load_raw_data(dataset_name, "processed_data.pt")
    train_data, base_dir = _load_raw_data(dataset_name, "processed_data_link_notest.pt")
    x_features = _load_simteg_features(base_dir, train_data.num_nodes)

    full_pairs = _edge_index_to_pairs(full_data.edge_index)
    train_candidate_pairs = _edge_index_to_pairs(train_data.edge_index)
    test_pairs = full_pairs - train_candidate_pairs
    train_pairs, val_pairs = _split_pairs(train_candidate_pairs)

    val_neg_pairs = _sample_negative_pairs(train_data.num_nodes, full_pairs, len(val_pairs), RANDOM_SEED + 1)
    test_neg_pairs = _sample_negative_pairs(train_data.num_nodes, full_pairs, len(test_pairs), RANDOM_SEED + 2)

    print(
        "LP split: "
        f"train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}, "
        f"val_neg={len(val_neg_pairs)}, test_neg={len(test_neg_pairs)}"
    )

    final_data = Data(
        x=x_features,
        edge_index=_pairs_to_edge_index(train_pairs, undirected=True),
        y=getattr(train_data, "y", None),
        num_nodes=x_features.shape[0],
    )
    final_data.val_pos_edge_index = _pairs_to_edge_index(val_pairs)
    final_data.val_neg_edge_index = _pairs_to_edge_index(val_neg_pairs)
    final_data.test_pos_edge_index = _pairs_to_edge_index(test_pairs)
    final_data.test_neg_edge_index = _pairs_to_edge_index(test_neg_pairs)
    final_data.all_pos_edge_index = _pairs_to_edge_index(full_pairs, undirected=True)
    return final_data
