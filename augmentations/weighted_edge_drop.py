from typing import Optional, Tuple

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor
from torch_geometric.utils import cumsum, degree, sort_edge_index, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


def centrality_weighted(edge_index: Tensor,
                 centrality: Tensor,
                 p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if not training:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

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