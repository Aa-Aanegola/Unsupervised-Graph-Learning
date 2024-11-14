import wandb
import pandas as pd
from datetime import datetime

# Set your project details
ENTITY = "aa_aanegola"  # replace with your wandb entity (usually your username)
PROJECT = "Unsup-GNN"  # replace with your wandb project name

# Define date range for filtering
START_DATE = "2024-11-13"  # format: "YYYY-MM-DDTHH:MM:SSZ"
# Initialize WandB API
api = wandb.Api()

# Fetch all runs from the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

# Convert the date strings to datetime objects
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")

# Create a list to hold each run's summary metrics
run_data = []

# Loop through each run, extract summary metrics, and filter by date
for run in runs:
    run_created_at = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
    if start_date <= run_created_at:
        run_summary = {}
        run_summary["Run ID"] = run.id
        run_summary["date"] = run.name.split(' ')[0]
        run_summary["Model"] = run.name.split(' ')[-1]
        run_summary["Experiment Type"] = run.name.split(' ')[2]
        run_summary['Dataset'] = run.name.split(' ')[1]
        run_summary['Centrality'] = run.name.split(' ')[3]
        run_summary.update(run.summary._json_dict)

        run_data.append(run_summary)

# Convert list of dictionaries to a pandas DataFrame
df = pd.DataFrame(run_data)

# Display the DataFrame
print(df)
df.to_csv('wandb_data.csv', index=False)