from pathlib import Path
import pandas  as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, UnivariateSpline




def read_in_both_types(sleep_type, name, path_to_data):
    first = pd.read_csv(path_to_data/ f"{sleep_type}_{name}").drop(["Unnamed: 0"],axis=1)
    #second = pd.read_csv(path_to_data/ f"non-{sleep_type}_{name}").drop(["Unnamed: 0"],axis=1)
    return pd.concat([first], axis=1).mean(axis=1)

def triangular(x, m, c, p, b):
    return m * np.abs((x - c) % p - p/2) + b

def main():
    path_to_data = Path(__file__).parent / "data" 
    path_to_figures = Path(__file__).parent / "figures" 
    brier_uncertainty = read_in_both_types("insmnia5", "brier_adam_std_predictions_LSTM.csv", path_to_data)
    hard_uncertainty = read_in_both_types("insmnia5", "hard_adam_std_predictions_LSTM.csv", path_to_data)
    soft_uncertainty = read_in_both_types("insmnia5", "soft_adam_std_predictions_LSTM.csv", path_to_data)
    #set_brier_uncertainty = read_in_both_types("insmnia5", "brier_super_adam_std_predictions.csv", path_to_data)
    set_uncertainty = read_in_both_types("insmnia5", "set_variance.csv", path_to_data)
    set_uncertainty_std = set_uncertainty**0.5
    times_based_on_minutes = [pd.Timestamp(f"00:00:00") + pd.Timedelta(seconds=thirdy_seconds) for thirdy_seconds in range(0,len(set_uncertainty_std)*30,30)]
    times_only = [t.time().strftime('%H:%M:%S') for t in times_based_on_minutes]

    fig , ax = plt.subplots(ncols=1, nrows=1, figsize=(20,13))
   
    data_y = set_uncertainty_std
    data_t = np.arange(len(data_y))
    cs = UnivariateSpline(data_t, data_y, k=4)
    fitted_curve = cs(data_t)
    ax.plot(times_only, data_y, label="ensemble std", color="darkblue", linewidth=2.0)
    ax.plot(times_only, fitted_curve, '--', label=f'Fitted Cosine', color='black', linewidth=2.5)
    hours = [time for time in times_only if time.endswith(':00:00')]
    hourly_times = [time for i, time in enumerate(hours) if i % 2 == 0]
    ax.set_xticks(hourly_times)  # setting the x-ticks positions
    ax.set_xticklabels(hourly_times, rotation=45, fontsize=16)
    A = (max(fitted_curve)-min(fitted_curve))/2
    offset = (max(fitted_curve)+min(fitted_curve))/2  
    ax.set_ylabel("Standard Deviation",fontsize=22)
    ax.tick_params(axis='x', labelsize=18)  # setting the x-tick labels font size
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel("Time during day", fontsize=22)
    ax.set_title(f"Ensemble uncertainty \nAmplitude: {np.round(A, decimals=4)} \nOffset: {np.round(offset, decimals=4)}",  loc='left', fontweight='bold', fontsize=24)
    ax.legend(loc="upper left")
    left, bottom, width, height = [0.45, 0.62, 0.35, 0.35] # Adjust these values as per your needs
    inset_ax = fig.add_axes([left, bottom, width, height])

    # Plot on inset
    inset_ax.plot(times_only, brier_uncertainty, label="brier score", color="#E63946") 
    inset_ax.plot(times_only, soft_uncertainty, label="soft labels", color="#2A9D8F")  
    inset_ax.plot(times_only, hard_uncertainty, label="hard labels", color="#F4A261") 
    #inset_ax.plot(times_only, set_brier_uncertainty, label="brier set labels", color="#6B5B95") 
    inset_ax.set_title("")
    inset_ax.set_xticks(hourly_times)  # setting the x-ticks positions
    inset_ax.tick_params(axis='x', labelsize=16)
    inset_ax.tick_params(axis='y', labelsize=16)
    inset_ax.set_xticklabels(hourly_times, rotation=45)
    inset_ax.legend(fontsize=18)
    plt.savefig(path_to_figures / "overall_uncertainty_comparison_LSTM.png")
    plt.show()



    # Plot all uncertainties in one plot in the first row
    all_uncertainties = [brier_uncertainty, hard_uncertainty, soft_uncertainty]
    all_uncertainties_labels = ["Brier Score Uncertainty", "Hard Label Uncertainty", "Soft Label Uncertainty"]
    fig, axes = plt.subplots(nrows=1, ncols=len(all_uncertainties), figsize=(15, 8), sharey='row')
    colors = ['#6B5B95', '#E63946', '#F4A261', '#2A9D8F']  # Assuming you have at most 6 different uncertainties
    for i, ax in enumerate(axes):
        data_y = all_uncertainties[i]
        data_t = np.arange(len(data_y))
        #params, _ = curve_fit(triangular, data_t, data_y, maxfev=10000, p0=[0.001, 0, 2500, 0.02])
        #A = params[0] * params[2] / 2
        #fitted_curve = triangular(data_t, *params)
        cs = UnivariateSpline(data_t, data_y, k=4)
        fitted_curve = cs(data_t)
        A = (max(fitted_curve)-min(fitted_curve))/2
        offset = (max(fitted_curve)+min(fitted_curve))/2
        ax.plot(times_only, data_y, label=all_uncertainties_labels[i], color=colors[i], linewidth=1)
        ax.plot(times_only, fitted_curve, '--', label=f'Fitted Cosine - {all_uncertainties_labels[i]}', color='black', linewidth=2.5)
        ax.set_title(f"{all_uncertainties_labels[i]} \n Amplitude: {np.round(A, decimals=4)} \n Offset: {np.round(offset, decimals=4)}", 
                     fontweight='bold', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', labelsize=12)  # setting the x-tick labels font size
        ax.tick_params(axis='y', labelsize=12)
        ax.set_facecolor('white')
        ax.set_xticks(hourly_times)  # setting the x-ticks positions
        ax.set_xticklabels(hourly_times, rotation=45)

    plt.tight_layout()

    # Save or display the plot
    plt.savefig(path_to_figures / "uncertainty_comparison_LSTM.png")
    plt.show()

if __name__ == "__main__":
    main()