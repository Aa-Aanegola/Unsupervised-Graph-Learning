# take in a dataset name and run for both the unweighted and weighted
python3 train.py --flagfile=./config/$1.cfg 
python3 train.py --flagfile=./config/weighted-edge-drop/$1.cfg
