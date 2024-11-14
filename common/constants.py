from enum import Enum
import torch

class TransformType(Enum):
    DROP_EDGE = 'drop_edge'
    DROP_EDGE_WEIGHTED = 'drop_edge_weighted'
    DROP_EDGE_EXTENDED = 'drop_edge_extended'
    DROP_EDGE_WEIGHTED_EXTENDED = 'drop_edge_weighted_extended'

device = 'cuda' if torch.cuda.is_available() else 'cpu'