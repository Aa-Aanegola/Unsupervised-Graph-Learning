# take in a dataset name and run all possible configurations
# $1: dataset name
python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$1.cfg --transform_type=drop_edge
python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$1.cfg --transform_type=drop_edge_weighted
python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$1.cfg --transform_type=drop_edge_extended
python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$1.cfg --transform_type=drop_edge_weighted_extended
