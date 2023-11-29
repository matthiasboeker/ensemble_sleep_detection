from pathlib import Path
import pandas  as pd 
import matplotlib.pyplot as plt

def main():
    path_to_data = Path(__file__).parent / "data" / "mental_health"
    path_to_figures = Path(__file__).parent / "figures" 

    control_soft_mean = pd.read_csv(path_to_data/ "control_soft_score_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)
    depression_soft_mean = pd.read_csv(path_to_data/ "depression_soft_score_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)
    schizophrenia_soft_mean = pd.read_csv(path_to_data/ "schizophrenia_soft_score_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)

    control_hard_mean = pd.read_csv(path_to_data/ "control_hard_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)
    depression_hard_mean = pd.read_csv(path_to_data/ "depression_hard_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)
    schizophrenia_hard_mean = pd.read_csv(path_to_data/ "schizophrenia_hard_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)

    control_brier_mean = pd.read_csv(path_to_data/ "control_brier_score_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)
    depression_brier_mean = pd.read_csv(path_to_data/ "depression_brier_score_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)
    schizophrenia_brier_mean = pd.read_csv(path_to_data/ "schizophrenia_brier_score_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1)

    control_soft_std = pd.read_csv(path_to_data/ "control_soft_score_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)
    depression_soft_std = pd.read_csv(path_to_data/ "depression_soft_score_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)
    schizophrenia_soft_std = pd.read_csv(path_to_data/ "schizophrenia_soft_score_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)

    control_hard_std = pd.read_csv(path_to_data/ "control_hard_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)
    depression_hard_std = pd.read_csv(path_to_data/ "depression_hard_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)
    schizophrenia_hard_std = pd.read_csv(path_to_data/ "schizophrenia_hard_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)

    control_brier_std = pd.read_csv(path_to_data/ "control_brier_score_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)
    depression_brier_std = pd.read_csv(path_to_data/ "depression_brier_score_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)
    schizophrenia_brier_std = pd.read_csv(path_to_data/ "schizophrenia_brier_score_adam_std_predictions.csv").drop(["Unnamed: 0"],axis=1)

    control_ensemble_std = pd.read_csv(path_to_data/ "control_set_variance.csv").drop(["Unnamed: 0"],axis=1).dropna()**0.5
    depression_ensemble_std = pd.read_csv(path_to_data/ "depression_set_variance.csv").drop(["Unnamed: 0"],axis=1).dropna()**0.5
    schizophrenia_ensemble_std = pd.read_csv(path_to_data/ "schizophrenia_set_variance.csv").drop(["Unnamed: 0"],axis=1).dropna()**0.5

    times_based_on_minutes = [pd.Timestamp(f"00:00") + pd.Timedelta(minutes=minutes) for minutes in range(0,len(schizophrenia_brier_std))]
    times_only = [t.time().strftime('%H:%M:%S') for t in times_based_on_minutes]
    hours = [time for time in times_only if time.endswith('00:00')]
    hourly_times = [time for i, time in enumerate(hours) if i % 2 == 0]



    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(15,10), sharey=True)

    # Soft Labels Mean
    ax1.plot(times_only, control_soft_mean.mean(axis=1), label="control", color=colors[0], linewidth=2.5)
    ax1.plot(times_only, depression_soft_mean.mean(axis=1), label="depression", color=colors[1], linewidth=2.5)
    ax1.plot(times_only, schizophrenia_soft_mean.mean(axis=1), label="schizophrenia", color=colors[2], linewidth=2.5)
    ax1.set_title("Soft Labels Mean", fontsize=16, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax1.set_ylabel("Value")
    ax1.set_xlabel("Index")
    ax1.set_xticks(hourly_times)

    # Hard Labels Mean
    ax2.plot(times_only, control_hard_mean.mean(axis=1), label="control", color=colors[0], linewidth=2.5)
    ax2.plot(times_only, depression_hard_mean.mean(axis=1), label="depression", color=colors[1], linewidth=2.5)
    ax2.plot(times_only, schizophrenia_hard_mean.mean(axis=1), label="schizophrenia", color=colors[2], linewidth=2.5)
    ax2.set_title("Hard Labels Mean", fontsize=16, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax2.set_ylabel("Value")
    ax2.set_xlabel("Index")
    ax2.set_xticks(hourly_times)

    # Brier Score
    ax3.plot(times_only, control_brier_mean.mean(axis=1), label="control", color=colors[0], linewidth=2.5)
    ax3.plot(times_only, depression_brier_mean.mean(axis=1), label="depression", color=colors[1], linewidth=2.5)
    ax3.plot(times_only, schizophrenia_brier_mean.mean(axis=1), label="schizophrenia", color=colors[2], linewidth=2.5)
    ax3.set_title("Brier Score Mean", fontsize=16, fontweight='bold')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax3.set_ylabel("Value")
    ax3.set_xlabel("Index")
    ax3.set_xticks(hourly_times)


    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the figure
    plt.savefig(path_to_figures / "mean_mental_health_soft.png", bbox_inches='tight')
    plt.show()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=1, nrows=4, figsize=(20,20),sharey=False)
    # Soft Labels Mean
    ax1.plot(times_only, control_soft_std.mean(axis=1), label="control", color=colors[0], linewidth=2.5)
    ax1.plot(times_only, depression_soft_std.mean(axis=1), label="depression", color=colors[1], linewidth=2.5)
    ax1.plot(times_only, schizophrenia_soft_std.mean(axis=1), label="schizophrenia", color=colors[2], linewidth=2.5)
    ax1.set_title("Soft Labels", fontsize=18, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=16)
    ax1.set_ylabel("Standard Deviation", fontsize=16)
    ax1.set_xlabel("Time of Day", fontsize=16)
    ax1.set_xticks(hourly_times)
    ax1.set_xticklabels(hourly_times, rotation=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    # Hard Labels Std
    ax2.plot(times_only, control_hard_std.mean(axis=1), label="control", color=colors[0], linewidth=2.5)
    ax2.plot(times_only, depression_hard_std.mean(axis=1), label="depression", color=colors[1], linewidth=2.5)
    ax2.plot(times_only, schizophrenia_hard_std.mean(axis=1), label="schizophrenia", color=colors[2], linewidth=2.5)
    ax2.set_title("Hard Labels Std", fontsize=18, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=16)
    ax2.set_ylabel("Standard Deviation", fontsize=16)
    ax2.set_xlabel("Time of Day", fontsize=16)
    ax2.set_xticks(hourly_times)
    ax2.set_xticklabels(hourly_times, rotation=20)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)

    # Brier Std
    ax3.plot(times_only, control_brier_std.mean(axis=1), label="control", color=colors[0], linewidth=2.5)
    ax3.plot(times_only, depression_brier_std.mean(axis=1), label="depression", color=colors[1], linewidth=2.5)
    ax3.plot(times_only, schizophrenia_brier_std.mean(axis=1), label="schizophrenia", color=colors[2], linewidth=2.5)
    ax3.set_title("Brier Score Std", fontsize=18, fontweight='bold')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=16)
    ax3.set_ylabel("Standard Deviation", fontsize=16)
    ax3.set_xlabel("Time of Day", fontsize=16)
    ax3.set_xticks(hourly_times)
    ax3.set_xticklabels(hourly_times, rotation=20)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)

    # 
    ax4.plot(times_only, control_ensemble_std.mean(axis=1), label="control", color=colors[0], linewidth=2.5)
    ax4.plot(times_only, depression_ensemble_std.mean(axis=1), label="depression", color=colors[1], linewidth=2.5)
    ax4.plot(times_only, schizophrenia_ensemble_std.mean(axis=1), label="schizophrenia", color=colors[2], linewidth=2.5)
    ax4.set_title("Ensemble Variance", fontsize=18, fontweight='bold')
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax4.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=16)
    ax4.set_ylabel("Standard Deviation", fontsize=16)
    ax4.set_xlabel("Time of Day", fontsize=16)
    ax4.set_xticks(hourly_times)
    ax4.set_xticklabels(hourly_times, rotation=20)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.tick_params(axis='y', labelsize=16)


    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the figure
    plt.savefig(path_to_figures / "std_mental_health_soft.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()