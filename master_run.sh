# Array of job names (tmux session names) and commands
declare -A jobs=(
    ["cora"]="./run.sh cora"
    ["citeseer"]="./run.sh citeseer"
    ["photos"]="./run.sh amazon-photos"
    ["computers"]="./run.sh amazon-computers"
    ["coauthor-cs"]="./run.sh coauthor-cs"
    ["coauthor-physics"]="./run.sh coauthor-physics"
)

for job in "${!jobs[@]}"; do
    # Create a new tmux session for each job
    tmux new-session -d -s "$job" # '-d' creates the session detached
    
    # Send the job command to the tmux session and run it in the background
    tmux send-keys -t "$job" "${jobs[$job]} &" C-m  # '&' to run in background, C-m to simulate Enter key
done