from torch_geometric.datasets import KarateClub, Planetoid
from weighted_edge_drop import *
from edge_add import *
import networkx as nx
from networkx.algorithms.centrality import (
    betweenness_centrality, 
    degree_centrality
)
from torch_geometric.utils import to_networkx
import torch
import pickle as pkl

dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

def log_normalize(d):
    arr = torch.tensor([v for k, v in sorted(d.items())])
    mn = min(arr)
    mx = max(arr)
    arr = (arr-mn)/(mx-mn)
    return arr

G = to_networkx(data, to_undirected=True)
print(G)
centrality = log_normalize(degree_centrality(G))
print(centrality)

edge_index = data.edge_index
new_edge_index, edge_mask = centrality_weighted(edge_index, centrality, p=0.5, force_undirected=True, training=True)
edge_add_index, edge_add_mask = two_hop_edge_dropout(edge_index, data.num_nodes, p=0.5, force_undirected=True, training=True)
edge_add_weighted_index, edge_add_weighted_mask = two_hop_centrality_weighted(edge_index, data.num_nodes, centrality, p=0.5, force_undirected=True, training=True)

print(edge_index.shape, edge_index)
print(new_edge_index.shape, new_edge_index)
print(edge_add_index.shape, edge_add_index)
print(edge_add_weighted_index.shape, edge_add_weighted_index)

edge_set = set([(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])])
new_edge_set = set([(new_edge_index[0, i].item(), new_edge_index[1, i].item()) for i in range(new_edge_index.shape[1])])
edge_add_set = set([(edge_add_index[0, i].item(), edge_add_index[1, i].item()) for i in range(edge_add_index.shape[1])])
edge_add_weighted_set = set([(edge_add_weighted_index[0, i].item(), edge_add_weighted_index[1, i].item()) for i in range(edge_add_weighted_index.shape[1])])