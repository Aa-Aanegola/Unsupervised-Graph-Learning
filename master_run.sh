#!/bin/bash

# Array of job names (dataset names) and configurations
declare -A jobs=(
    ["cora"]="cora"
    ["citeseer"]="citeseer"
    ["photos"]="amazon-photos"
    ["computers"]="amazon-computers"
    ["twitch-en"]="twitch-en"
    ["twitch-de"]="twitch-de"
)

# Array of centrality types
centralities=("degree_centrality.pkl" "eigen_centrality.pkl" "betweenness_centrality.pkl")

for job in "${!jobs[@]}"; do
    dataset="${jobs[$job]}"
    echo "Running configurations for dataset: $dataset"
    
    for centrality in "${centralities[@]}"; do
        centrality_path="$centrality"
        
        # Run all configurations for the current dataset and centrality type
        python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$dataset.cfg --transform_type=drop_edge --centrality_path=$centrality_path
        python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$dataset.cfg --transform_type=drop_edge_weighted --centrality_path=$centrality_path
        python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$dataset.cfg --transform_type=drop_edge_extended --centrality_path=$centrality_path
        python3 train.py --flagfile=/home/aa-aanegola/Unsupervised-Graph-Learning/config/$dataset.cfg --transform_type=drop_edge_weighted_extended --centrality_path=$centrality_path
    done
done
