import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def to_pyg_data(x_features: np.ndarray, adj_matrix: np.ndarray) -> Data:
    """Convert dense features and an adjacency matrix to a PyG Data object."""
    from scipy.sparse import csr_matrix

    x = torch.tensor(x_features, dtype=torch.float)
    adj_coo = csr_matrix(adj_matrix).tocoo()
    edge_index = torch.tensor(np.array([adj_coo.row, adj_coo.col]), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class GcnLayers(nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout):
        super().__init__()
        self.act = nn.ELU()
        self.num_layers_num = num_layers_num
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        for layer_idx in range(num_layers_num):
            in_dim = n_in if layer_idx == 0 else n_h
            self.convs.append(GCNConv(in_dim, n_h, normalize=True))
            self.bns.append(nn.BatchNorm1d(n_h))

    def forward(self, x, edge_index):
        graph_output = None
        for layer_idx, conv in enumerate(self.convs):
            if layer_idx == 0:
                graph_output = conv(x, edge_index)
            else:
                graph_output = conv(graph_output, edge_index) + graph_output
            graph_output = self.act(graph_output)
        return graph_output


def pca_compression(features, k):
    pca = PCA(n_components=k)
    compressed = pca.fit_transform(features)
    print(f"PCA retained variance ratio: {pca.explained_variance_ratio_.sum():.6f}")
    return compressed


def svd_compression(features, k):
    u, sigma, _ = np.linalg.svd(features)
    return u[:, :k].dot(np.diag(sigma[:k]))


class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout, sample=None):
        super().__init__()
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        if sample is not None:
            self.register_buffer("negative_sample", torch.tensor(sample, dtype=torch.int64))
        else:
            self.negative_sample = None
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        g = self.gcn(x, edge_index)
        return compareloss(g, self.negative_sample, temperature=1)

    def embed(self, x, edge_index):
        return self.gcn(x, edge_index).detach()


def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten().reshape(-1, 1)
    index = torch.broadcast_to(index, (index.size(0), feature.size(1)))
    gathered = torch.gather(feature, dim=0, index=index)
    return gathered.reshape(input_size, -1, feature.size(1))


def compareloss(feature, tuples, temperature):
    tuples = tuples.to(feature.device)
    h_tuples = mygather(feature, tuples)

    base_index = torch.arange(0, len(tuples), device=feature.device).reshape(-1, 1)
    base_index = torch.broadcast_to(base_index, (base_index.size(0), tuples.size(1)))
    h_i = mygather(feature, base_index)

    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:].permute(1, 0).sum(dim=1, keepdim=True)
    return (-torch.log(numerator / denominator)).mean()


def prompt_pretrain_sample(edge_index, n):
    node_count = edge_index.max().item() + 1
    adj_dict = {idx: set() for idx in range(node_count)}
    for src, dst in edge_index.T.tolist():
        adj_dict[src].add(dst)
        adj_dict[dst].add(src)

    samples = np.zeros((node_count, 1 + n), dtype=int)
    all_nodes = np.arange(node_count)
    for idx in range(node_count):
        neighbors = list(adj_dict[idx])
        non_neighbors = np.setdiff1d(all_nodes, neighbors)
        samples[idx][0] = idx if len(neighbors) == 0 else neighbors[0]
        np.random.shuffle(non_neighbors)
        samples[idx][1:1 + n] = non_neighbors[:n]

    return samples
