import copy
import sys
sys.path.append('../..')
from augmentations import *
from common.constants import TransformType

import torch
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.transforms import Compose

class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        # edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)
        edge_index, edge_attr = dropout_edge(edge_index, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

# use the weighted_edge_drop
class DropEdgesWeighted:
    r"""Drops edges with probability p after weighting by centrality."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = centrality_weighted(edge_index, data.centrality, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

class DropEdgesExtended:
    r"""Drops edges with probability p after adding two-hop edges to the graph."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None
        two_hop_edges = data.two_hop_edges if 'two_hop_edges' in data else None

        edge_index, edge_attr = two_hop_edge_dropout(edge_index, two_hop_edges, data.num_nodes, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

class DropEdgesWeightedExtended:
    r"""Drops edges with probability p after weighting by centrality and adding two-hop edges to the graph."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None
        two_hop_edges = data.two_hop_edges if 'two_hop_edges' in data else None

        edge_index, edge_attr = two_hop_centrality_weighted(edge_index, two_hop_edges, data.num_nodes, data.centrality, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

def get_graph_drop_transform(drop_edge_p, drop_feat_p, FLAGS):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    if FLAGS.transform_type == TransformType.DROP_EDGE.value:
        transforms.append(DropEdges(drop_edge_p))
    elif FLAGS.transform_type == TransformType.DROP_EDGE_WEIGHTED.value:
        transforms.append(DropEdgesWeighted(drop_edge_p))
    elif FLAGS.transform_type == TransformType.DROP_EDGE_EXTENDED.value:
        transforms.append(DropEdgesExtended(drop_edge_p))
    elif FLAGS.transform_type == TransformType.DROP_EDGE_WEIGHTED_EXTENDED.value:
        transforms.append(DropEdgesWeightedExtended(drop_edge_p))
    else:
        raise ValueError('Invalid transform type %s' % FLAGS.transform_type)

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)