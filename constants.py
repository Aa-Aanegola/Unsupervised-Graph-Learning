from enum import Enum

class TransformType(Enum):
    DROP_EDGE = 'drop_edge'
    DROP_EDGE_WEIGHTED = 'drop_edge_weighted'
    DROP_EDGE_EXTENDED = 'drop_edge_extended'
    DROP_EDGE_WEIGHTED_EXTENDED = 'drop_edge_weighted_extended'