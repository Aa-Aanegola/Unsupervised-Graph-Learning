import torch
from torch import Tensor

from torch_geometric.datasets import Amazon, Planetoid, Coauthor, Twitch


def sparse_two_hop_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    adj = torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.size(1)), 
        (num_nodes, num_nodes)
    )
    two_hop_adj = torch.sparse.mm(adj, adj).coalesce()
    two_hop_adj = two_hop_adj - adj 
    two_hop_adj = two_hop_adj.coalesce()
    two_hop_edges = two_hop_adj.indices()

    # remove self edges
    mask = two_hop_edges[0] != two_hop_edges[1]
    two_hop_edges = two_hop_edges[:, mask]

    return two_hop_edges

dset = 'Computers'
dataset = Amazon(root='../data/', name=dset)
data = dataset[0]


two_hop_edges = sparse_two_hop_edges(data.edge_index, data.num_nodes)
print(two_hop_edges.shape)
torch.save(two_hop_edges, f'../data/{dset}/two_hop_edges.pt')