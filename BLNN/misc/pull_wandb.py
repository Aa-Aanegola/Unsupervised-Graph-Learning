import wandb
import pandas as pd
from datetime import datetime

def pull_data():
    # Set your project details
    ENTITY = "aa_aanegola"  # replace with your wandb entity (usually your username)
    PROJECT = "Unsup-GNN"  # replace with your wandb project name

    # Define date range for filtering
    START_DATE = "2024-11-23"  # format: "YYYY-MM-DDTHH:MM:SSZ"
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

# read from csv file 
df = pd.read_csv('wandb_data.csv')

# Important columns - Model, Experiment Type, Dataset, Centrality, accuracy, accuracy_std, f1_score, f1_score_std, demographic_parity_ratio, demographic_parity_ratio_std, equal_opportunity_ratio, equal_opportunity_ratio_std, equalized_odds_ratio, equalized_odds_ratio_std, consistency, consistency_std, fairness, fairness_std, overall_accuracy, overall_accuracy_std, overall_f1_score, overall_f1_score_std, overall_demographic_parity_ratio, overall_demographic_parity_ratio_std, overall_equal_opportunity_ratio, overall_equal_opportunity_ratio_std, Sim@5, Sim@10, Homo, NMI
# Define the experiment types
experiment_types = ["drop_edge_weighted", "drop_edge_extended", "drop_edge_weighted_extended"]
baseline_type = "drop_edge"


exp_map = {
    "drop_edge_weighted": "CD", 
    "drop_edge_extended": "SC", 
    "drop_edge_weighted_extended": "SC+CD"
}

dataset_map = {
    "amazon-computers": "amznC", 
    "amazon-photos": "amznP", 
    "cora": "Cora", 
    "citeseer": "cseer", 
    "twitch-de": "twide",
    "twitch-en": "twien" 
}

# Initialize an empty list to hold the comparison results
comparison_results = []

# Group the DataFrame by Model, Experiment Type, Dataset, and Centrality
grouped = df.groupby(["Model", "Dataset", "Centrality"])

# Loop through each group
for (model, dataset, centrality), group in grouped:
    # Get the baseline metrics
    baseline = group[group["Experiment Type"] == baseline_type]
    if baseline.empty:
        continue
    baseline_metrics = baseline.iloc[0]

    # Loop through each experiment type
    for exp_type in experiment_types:
        exp = group[group["Experiment Type"] == exp_type]
        if exp.empty:
            continue
        exp_metrics = exp.iloc[0]

        # Calculate the percentage improvement for each metric
        comparison = {
            "Model": model,
            "Dataset": dataset_map[dataset],
            "Centrality": centrality[:3],
            "Experiment Type": exp_map[exp_type]
        }
        for col in df.columns:
            if col in ["accuracy", "f1_score", "demographic_parity_ratio", "Sim@5", "Sim@10", "Homo"]:
                baseline_value = baseline_metrics[col]
                exp_value = exp_metrics[col]
                if baseline_value != 0:
                    comparison[col] = round(((exp_value - baseline_value) / baseline_value) * 100, 2)
                else:
                    comparison[col] = None

        comparison_results.append(comparison)

# Convert the comparison results to a DataFrame
comparison_df = pd.DataFrame(comparison_results)
# rename the column headers 
comparison_df.rename(columns={
    "Model": "Model",
    "Dataset": "Dset",
    "Centrality": "Cent",
    "Experiment Type": "Exp",
    "accuracy": "Acc",
    "f1_score": "F1",
    "demographic_parity_ratio": "DPR",
    "Sim@5": "Sim5",
    "Sim@10": "Sim10",
    "Homo": "Homo"
}, inplace=True)
comparison_df = comparison_df[["Model", "Dset", "Cent", "Exp", "Acc", "F1", "DPR", "Sim5", "Sim10", "Homo"]]

# Display the comparison DataFrame
print(comparison_df)
comparison_df.to_csv('wandb_comparison.csv', index=False)
# Get unique models
models = comparison_df["Model"].unique()

# Loop through each model and save the corresponding DataFrame to a CSV file
for model in models:
    model_df = comparison_df[comparison_df["Model"] == model]
    model_df.to_csv(f'wandb_comparison_{model}.csv', index=False)