import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


def plot_metrics(df, metrics=["mcc", "kappa", "f1", "acc"],
             algorithms=["CK", "Sadeh", "HMM", "Sazonova", "Oakley", "Ensemble"]):
    # Define the number of subplots
    num_metrics = len(metrics)

    # Create a figure for the subplots
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    for i, metric in enumerate(metrics):
        # Extract the columns related to the current metric for each algorithm
        metric_columns = [f"{algo}_{metric}" for algo in algorithms if f"{algo}_{metric}" in df.columns]

        # Extract the relevant data from the DataFrame
        metric_data = df[metric_columns]

        # Create a boxplot for the current metric
        axs[i].boxplot(metric_data.values, labels=metric_columns, vert=False)
        axs[i].set_title(f"Comparison of Algorithms for {metric.upper()}")
        axs[i].set_xlabel(f"{metric.upper()} Value")
        axs[i].set_ylabel("Algorithms")

    # Show the plots
    plt.tight_layout()
    plt.show()


def main():
    path_to_data = Path(__file__).parent / "data" / "varibility_df.csv"
    df = pd.read_csv(path_to_data)
    print(df.head())
    for column in df:
        print(f"Mean of {column}: {df[column].mean()}")
        print(f"Variance of {column}: {df[column].std()}")
        
    plot_metrics(df, metrics=["mcc", "kappa", "f1", "acc"],
                 algorithms=["CK", "Sadeh", "HMM", "Sazonova", "Oakley", "Ensemble"])


if __name__ == "__main__":
    main()