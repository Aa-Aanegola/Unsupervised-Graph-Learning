#!/bin/bash

# Array of job names (dataset names) and configurations
declare -A jobs=(
    ["cora"]="cora"
    # ["citeseer"]="citeseer"
    # ["photos"]="amazon-photos"
    # ["computers"]="amazon-computers"
)

# Array of centrality types
centralities=("degree_centrality.pkl" "eigen_centrality.pkl")

for job in "${!jobs[@]}"; do
    dataset="${jobs[$job]}"
    echo "Running configurations for dataset: $dataset"
    
    for centrality in "${centralities[@]}"; do
        centrality_path="$centrality"
        
        # Run all configurations for the current dataset and centrality type
        # run each configuration 10 times
        # for i in {1..10}; do
        #     python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/BLNN/config/$dataset.cfg --transform_type=drop_edge --centrality_path=$centrality_path
        # done

        # for i in {1..10}; do
        #     python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/BLNN/config/$dataset.cfg --transform_type=drop_edge_weighted --centrality_path=$centrality_path
        # done

        # for i in {1..10}; do
        #     python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/BLNN/config/$dataset.cfg --transform_type=drop_edge_extended --centrality_path=$centrality_path --sample_two_hop=True
        # done

        # for i in {1..10}; do
        #     python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/BLNN/config/$dataset.cfg --transform_type=drop_edge_weighted_extended --centrality_path=$centrality_path --sample_two_hop=True
        # done

        for i in {1..10}; do
            python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/BLNN/config/$dataset.cfg --transform_type=drop_edge_extended --centrality_path=$centrality_path
        done

        for i in {1..10}; do
            python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/BLNN/config/$dataset.cfg --transform_type=drop_edge_weighted_extended --centrality_path=$centrality_path
        done
    done
done
