from typing import Optional, Tuple
import sys
sys.path.append('..')

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor
from torch_geometric.utils import cumsum, degree, sort_edge_index, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.loader import NeighborSampler
from common.constants import *

def sample_two_hop_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    sampler = NeighborSampler(edge_index, sizes=[1, 1])
    res = sampler.sample(batch=torch.arange(num_nodes))[2]
    two_hop_edges = torch.cat([torch.unsqueeze(res[0].edge_index[0], -1), torch.unsqueeze(res[1].edge_index[0], -1)], dim=-1)
    two_hop_edges = two_hop_edges[two_hop_edges[:, 0] != two_hop_edges[:, 1]].to(device)
    return two_hop_edges

def two_hop_edge_dropout(edge_index: Tensor, two_hop_edges: Tensor, num_nodes: int, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask
    
    if two_hop_edges == None:
        sample = sample_two_hop_edges(edge_index, num_nodes).T
    else:
        sample = two_hop_edges[:, torch.randperm(two_hop_edges.size(1))[:edge_index.size(1) // 5]]
    edge_index = torch.cat([edge_index, sample], dim=-1)

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def two_hop_centrality_weighted(edge_index: Tensor,
                 two_hop_edges: Tensor,
                 num_nodes: int,
                 centrality: Tensor,
                 p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    
    if not training:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask
    

    if two_hop_edges == None:
        sample = sample_two_hop_edges(edge_index, num_nodes).T
    else:
        sample = two_hop_edges[:, torch.randperm(two_hop_edges.size(1))[:edge_index.size(1) // 5]]
    edge_index = torch.cat([edge_index, sample], dim=-1)
    
    row, col = edge_index

    score = (centrality[row] + centrality[col] / 2) + 1
    score = score / torch.sum(score)
    idx = torch.multinomial(score, int(score.size(0) * p), replacement=False)
    edge_mask = torch.ones_like(score, dtype=torch.bool)
    edge_mask[idx] = False

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask