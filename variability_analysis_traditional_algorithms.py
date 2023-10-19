import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import json

from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, f1_score, balanced_accuracy_score, \
    brier_score_loss
from algorithms.bayesian_net import BayesianNet, BayesianCNN, BayesianLSTM
from sklearn.model_selection import train_test_split
from algorithms.uncertainty_measures import ar_covariance_estimation, compute_ece
from algorithms.loss_function import SoftCrossEntropy, BrierScoreLoss, SupersetBrierScoreLoss


def majority_vote_preds(predictions):
    return np.array([np.bincount(column).argmax() for column in np.array(predictions).T])


def write_dict_to_json(data, file_name, pretty_format=True):
    with open(file_name, 'w') as json_file:
        if pretty_format:
            json.dump(data, json_file, indent=4, sort_keys=True)
        else:
            json.dump(data, json_file)


def majority_vote(df):
    majority = df.mode(axis=1).iloc[:, 0]
    return majority


def normalise(series):
    return (series-series.mean())/series.std()


def switch_ones_and_zeros(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: 1 if x == 0 else 0)
    return df


def create_dataset(df):
    columns_to_switch = ["CK", "Sadeh", "HMM", "Sazonova", "Oakley"]
    if not any(column in df.columns for column in columns_to_switch):
        return None
    df = switch_ones_and_zeros(df, columns_to_switch)
    df["majority_vote"] = majority_vote(df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]])
    df["probability"] = df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]].mean(axis=1)
    return df


def get_metrics(gt, hard_predictions, name,  window_size):
    results = {}
    results["mcc"] = matthews_corrcoef(gt, hard_predictions[window_size:])
    results["kappa"] = cohen_kappa_score(gt, hard_predictions[window_size:])
    results["f1"] = f1_score(gt, hard_predictions[window_size:])
    results["acc"] = balanced_accuracy_score(gt, hard_predictions[window_size:])
    return results


def get_means(data):
    means = {}
    for alg, values in data.items():
        mean_mcc = np.nanmean([d['mcc'] for d in values])
        mean_kappa = np.nanmean([d['kappa'] for d in values])
        mean_f1 = np.nanmean([d['f1'] for d in values])
        mean_acc = np.nanmean([d['acc'] for d in values])

        means[alg] = {'mcc': mean_mcc, 'kappa': mean_kappa, 'f1': mean_f1, 'acc': mean_acc}
    return means


def get_acc_files(path_to_acc_files):
    return os.listdir(path_to_acc_files)


def main():
    path_to_data = Path(__file__).parent / "data" / "ensemble_files"
    path_to_save_res = Path(__file__).parent / "data"

    window_size = 20
    file_names = get_acc_files(path_to_data)
    n_iterations = 1000  # or any other number of iterations you want
    bootstrap_size = 50  # or you can choose another size
    bootstraps = []
    for i in range(n_iterations):
        print(f"Iteration {i + 1}/{n_iterations}")

        bootstrap_files = np.random.choice(file_names, size=bootstrap_size, replace=True)
        bootstrap_means = {}
        bootstrap_means["CK"] = []
        bootstrap_means["HMM"] = []
        bootstrap_means["Sadeh"] = []
        bootstrap_means["Sazonova"] = []
        bootstrap_means["Oakley"] = []
        bootstrap_means["Ensemble"] = []
        for file_name in bootstrap_files:
            acc_file = pd.read_csv(path_to_data / file_name, engine="python")
            df = create_dataset(acc_file)
            if df is None:
                continue
            last_valid_index = df["ground_truth"].replace(0, pd.NA).last_valid_index()
            if last_valid_index is None:
                nan_dict = {'mcc': np.nan, 'kappa': np.nan, 'f1': np.nan, 'acc': np.nan}
                bootstrap_means["CK"].append(nan_dict)
                bootstrap_means["HMM"].append(nan_dict)
                bootstrap_means["Sadeh"].append(nan_dict)
                bootstrap_means["Sazonova"].append(nan_dict)
                bootstrap_means["Oakley"].append(nan_dict)
                bootstrap_means["Ensemble"].append(nan_dict)
            else:
                test = df.loc[:last_valid_index + 2 * 60 * 6, :]
                ground_truth = test.ground_truth[window_size:].apply(lambda x: 1 if x > 0 else 0)
                bootstrap_means["CK"].append(get_metrics(ground_truth, test["CK"], "CK", window_size))
                bootstrap_means["HMM"].append(get_metrics(ground_truth, test["HMM"], "HMM", window_size))
                bootstrap_means["Sadeh"].append(get_metrics(ground_truth, test["Sadeh"], "Sadeh", window_size))
                bootstrap_means["Sazonova"].append(get_metrics(ground_truth, test["Sazonova"], "Sazonova", window_size))
                bootstrap_means["Oakley"].append(get_metrics(ground_truth, test["Oakley"], "Oakley", window_size))
                bootstrap_means["Ensemble"].append(get_metrics(ground_truth, test['majority_vote'], "Ensemble", window_size))
        bootstraps.append(get_means(bootstrap_means))
    algos = []
    for algo in ["CK", "HMM", "Sadeh", "Sazonova", "Oakley", "Ensemble"]:
        algo_df = pd.DataFrame([bootstrap[algo] for bootstrap in bootstraps])
        algo_df = algo_df.add_prefix(f"{algo}_", axis=1)
        algos.append(algo_df)
    merged_df = pd.concat(algos, axis=1)
    merged_df.to_csv("varibility_df.csv")

if __name__ == "__main__":
    main()