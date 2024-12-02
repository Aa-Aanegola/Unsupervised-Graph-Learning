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
            run_summary["Experiment Type"] = run.name.split(' ')[2] + (' all' if 'all' in run.name else '')
            run_summary['Dataset'] = run.name.split(' ')[1]
            run_summary['Centrality'] = run.name.split(' ')[3] if run.name.split(' ')[3] != 'all' else run.name.split(' ')[4]
            run_summary.update(run.summary._json_dict)

            run_data.append(run_summary)

    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(run_data)

    # Display the DataFrame
    # print(df)

    return df

data = pull_data()

# compute average statistics over runs with the same experiment type, dataset, centrality and model. Also compute the standard deviation
res = data.groupby(['Experiment Type', 'Dataset', 'Centrality', 'Model']).agg({'accuracy': ['mean', 'std'], 'f1_score': ['mean', 'std'], 'demographic_parity_ratio': ['mean', 'std'], 'Sim@5': ['mean', 'std'], 'Sim@10': ['mean', 'std']})
res = res.sort_values(by=['Model', 'Dataset', 'Centrality', 'Experiment Type'])
# for every model, dataset and centrality compute the improvement over the drop_edge baseline

improvements = []
for (model, dataset, centrality), group in res.groupby(level=['Model', 'Dataset', 'Centrality']):
    if 'drop_edge' in group.index.get_level_values('Experiment Type'):
        drop_edge_metrics = group.loc['drop_edge']
        for exp_type in group.index.get_level_values('Experiment Type'):
            if exp_type != 'drop_edge':
                improvement = {
                    'Experiment Type': exp_type,
                    'Dataset': dataset,
                    'Centrality': centrality,
                    'Model': model,
                    'accuracy': (group.loc[exp_type]['accuracy']['mean'].values[0] - drop_edge_metrics['accuracy']['mean'].values[0]) / drop_edge_metrics['accuracy']['mean'].values[0],
                    'f1_score': (group.loc[exp_type]['f1_score']['mean'].values[0] - drop_edge_metrics['f1_score']['mean'].values[0]) / drop_edge_metrics['f1_score']['mean'].values[0],
                    'demographic_parity_ratio': (group.loc[exp_type]['demographic_parity_ratio']['mean'].values[0] - drop_edge_metrics['demographic_parity_ratio']['mean'].values[0]) / drop_edge_metrics['demographic_parity_ratio']['mean'].values[0], 
                    'Sim@5': (group.loc[exp_type]['Sim@5']['mean'].values[0] - drop_edge_metrics['Sim@5']['mean'].values[0]) / drop_edge_metrics['Sim@5']['mean'].values[0],
                    'Sim@10': (group.loc[exp_type]['Sim@10']['mean'].values[0] - drop_edge_metrics['Sim@10']['mean'].values[0]) / drop_edge_metrics['Sim@10']['mean'].values[0]
                }
                improvements.append(improvement)


def improvement(x):
    return x.mean()

improvement_df = pd.DataFrame(improvements).groupby(['Model', 'Dataset', 'Centrality', 'Experiment Type']).agg({'accuracy': [improvement], 'f1_score': [improvement], 'demographic_parity_ratio': [improvement], 'Sim@5': [improvement], 'Sim@10': [improvement]})

# Merge the dataframes
res = res.reset_index()
res = res.merge(improvement_df, on=['Model', 'Dataset', 'Centrality', 'Experiment Type'])

res = res[['Model', 'Dataset', 'Centrality', 'Experiment Type', 
           'accuracy', 'f1_score', 'demographic_parity_ratio', 'Sim@5', 'Sim@10']]

res.sort_values(by=['Model', 'Dataset', 'Centrality', 'Experiment Type'], inplace=True)

# Save the results to a CSV file
res.to_csv('average_stats.csv', index=False)
pd.DataFrame(improvements).to_csv('improvements.csv', index=False)