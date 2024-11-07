# take in a dataset name and run all possible configurations
# $1: dataset name
python3 train.py --flagfile=./config/$1.cfg --transform_type=drop_edge
python3 train.py --flagfile=./config/$1.cfg --transform_type=weighted_edge_drop
python3 train.py --flagfile=./config/$1.cfg --transform_type=drop_edge_extended
python3 train.py --flagfile=./config/$1.cfg --transform_type=drop_edge_weighted_extended
