import warnings
import argparse
import json
from torch_geometric.datasets import Planetoid

import networkx as nx
from networkx.algorithms.centrality import (
    degree_centrality,
    eigenvector_centrality,
    betweenness_centrality
)
from torch_geometric.utils import to_networkx
import torch
import pickle as pkl

dataset = Planetoid(root='./data/', name='Cora')
data = dataset[0]

def log_normalize(d):
    arr = torch.tensor([v for k, v in sorted(d.items())])
    mn = min(arr)
    mx = max(arr)
    arr = (arr-mn)/(mx-mn)
    return arr

G = to_networkx(data, to_undirected=True)  #
print(G)

print(f"Starting Centrality calculation", flush=True)
degc = log_normalize(degree_centrality(G))
eigc = log_normalize(eigenvector_centrality(G))
betc = log_normalize(betweenness_centrality(G))
degree = torch.tensor([d for n, d in G.degree()]).float()
group = degree > torch.median(degree)
print(f"Finished Centrality calculation\n", flush=True)

with open(f"./data/Cora/degree_centrality.pkl", "wb") as f:
    pkl.dump(degc, f)

with open(f"./data/Cora/eigen_centrality.pkl", "wb") as f:
    pkl.dump(eigc, f)

with open(f'./data/Cora/betweenness_centrality.pkl', "wb") as f:
    pkl.dump(betc, f)

with open(f'./data/Cora/group.pkl', "wb") as f:
    pkl.dump(group, f)