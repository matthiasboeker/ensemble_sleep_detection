import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import json
import torch.nn as nn

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


def binary_entropy(p):
    """Calculate the entropy for a binary classification probability."""
    # Handle edge cases
    if p == 0 or p == 1:
        return 0
    return - (p * np.log(p) + (1-p) * np.log(1-p))


def overall_entropy(probabilities):
    """Calculate the overall entropy for a time series of predicted probabilities."""
    entropies = [binary_entropy(p) for p in probabilities]
    return np.mean(entropies)


def switch_ones_and_zeros(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: 1 if x == 0 else 0)
    return df


def create_dataset(df):
    columns_to_switch = ["CK", "Sadeh", "HMM", "Sazonova", "Oakley", "wake"]
    df = switch_ones_and_zeros(df, columns_to_switch)
    df["majority_vote"] = majority_vote(df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]])
    df["probability"] = df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]].mean(axis=1)
    return df


def transform_to_xy_tensor(X, labels_hard, labels_soft, labels_superset, window_size):
    X_data, Y_hard, Y_soft, Y_superset = [], [], [], []

    for i in range(len(X) - window_size):
        X_data.append(X[i:i + window_size])
        Y_hard.append(labels_hard[i + window_size])
        Y_soft.append(labels_soft[i + window_size])
        Y_superset.append(labels_superset[i + window_size])

    # Convert lists of arrays to single numpy arrays
    X_data = np.array(X_data)
    Y_hard = np.array(Y_hard)
    Y_soft = np.array(Y_soft)
    Y_superset = np.array(Y_superset)

    return torch.tensor(X_data).float(), torch.tensor(Y_hard).float(), torch.tensor(Y_soft).float(), torch.tensor(Y_superset).float()


def get_metrics(gt, hard_predictions, name,  window_size):
    results = {}
    results["mcc"] = matthews_corrcoef(gt, hard_predictions[window_size:])
    results["kappa"] = cohen_kappa_score(gt, hard_predictions[window_size:])
    results["f1"] = f1_score(gt, hard_predictions[window_size:])
    results["acc"] = balanced_accuracy_score(gt, hard_predictions[window_size:])
    results["name"] = name
    return results


def train(model, optimizer, criterion, X, Y, epochs, batch_size, patience, min_delta):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # Create DataLoader for test set
    test_data = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    best_test_loss = float('inf')  # initialize with a high value
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        average_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                test_outputs = model(inputs)
                loss = criterion(test_outputs.squeeze(), labels)
                test_loss += loss.item() * inputs.size(0)

        average_test_loss = test_loss / len(test_loader.dataset)

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_train_loss}, Test Loss: {average_test_loss}")

        # Check for improvement
        if average_test_loss < best_test_loss - min_delta:
            best_test_loss = average_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break


def get_acc_files(path_to_acc_files):
    return os.listdir(path_to_acc_files)


def update_aggregated_metrics(aggregated, ref_name, metrics):
    if ref_name not in aggregated:
        aggregated[ref_name] = {}
    for key, value in metrics.items():
        # Check if the metric value is numeric
        if isinstance(value, (int, float)):
            if key not in aggregated[ref_name]:
                aggregated[ref_name][key] = {'sum': 0, 'count': 0}
            aggregated[ref_name][key]['sum'] += value
            aggregated[ref_name][key]['count'] += 1
            # Ensure that the 'values' key exists
            if 'values' not in aggregated[ref_name][key]:
                aggregated[ref_name][key]['values'] = []
            aggregated[ref_name][key]['values'].append(value)


def calculate_mean_metrics(aggregated_metrics):
    mean_metrics = {}
    variance_metrics = {}

    for setup, metrics_data in aggregated_metrics.items():
        mean_metrics[setup] = {}
        variance_metrics[setup] = {}

        for metric_name, metric_values in metrics_data.items():
            mean_value = metric_values["sum"] / metric_values["count"]

            # Compute variance
            variance = sum([(value - mean_value) ** 2 for value in metric_values["values"]]) / metric_values["count"]

            mean_metrics[setup][metric_name] = mean_value
            variance_metrics[setup][f"{metric_name}_variance"] = variance

    return {"Mean": mean_metrics, "Variance": variance_metrics}


def run_training(X_train, Y_train, classifier_name, classifier, criterion, optimizer, epochs, batch_size, patience=20, min_delta=0.001):
    if classifier_name == "LSTM":
        X_train = X_train.unsqueeze(2)
    train(classifier, optimizer, criterion, X_train, Y_train, epochs, batch_size, patience, min_delta)


def run_testing(X_test, Y_test, gt, classifier_name, classifier, samples):
    results = {}
    classifier.train()
    if classifier_name == "LSTM":
        X_test = X_test.unsqueeze(2)

    predictions = []
    for _ in range(samples):
        with torch.no_grad():
            outputs = classifier(X_test)
            predictions.append(outputs.squeeze())
    predictions = torch.stack(predictions)

    mean_predictions = predictions.mean(axis=0)
    hard_predictions = torch.round(mean_predictions, decimals=0)
    #plt.plot(Y_test, label="ensemble")
    #plt.plot(hard_predictions, label="hard")
    #plt.plot(gt, label="gt")
    #plt.legend()
    plt.show()
    results["mcc"] = matthews_corrcoef(gt, hard_predictions)
    results["kappa"] = cohen_kappa_score(gt, hard_predictions)
    results["f1"] = f1_score(gt, hard_predictions)
    results["acc"] = balanced_accuracy_score(gt, hard_predictions)
    results["brier"] = brier_score_loss(gt, mean_predictions)
    results["ece"] = compute_ece(mean_predictions, gt, [0.25, 0.5, 0.75, 1.0]).cpu().tolist()
    results["entropy"] = overall_entropy(mean_predictions).tolist()
    results["variance"] = ar_covariance_estimation(predictions).cpu().numpy().tolist()
    results["name"] = classifier_name
    return results, predictions


def main():
    path_to_data = Path(__file__).parent / "data" / "ensemble_files"
    path_to_save_res = Path(__file__).parent / "data"

    window_size = 20
    models = {"NN": BayesianNet(window_size=window_size, dropout=0.5, temperature=2),
              #"LSTM": BayesianLSTM(input_dim=1, hidden_dim=50, dropout=0.5, num_layers=2),
              #"CNN": BayesianCNN(window_size=window_size, dropout=0.5),
              }
    file_names = get_acc_files(path_to_data)
    aggregated_metrics = {}
    trad_aggregated = {}
    for i, file_name in enumerate(file_names):
        print(f"Processing: {file_name} Nr. {i}")
        acc_file = pd.read_csv(path_to_data / file_name, engine="python")
        df = create_dataset(acc_file)
        test = df.loc[:df["ground_truth"].replace(0, pd.NA).last_valid_index()+2*60*6,:]
        ground_truth = test.ground_truth[window_size:].apply(lambda x: 1 if x > 0 else 0)
        train = df.loc[df["ground_truth"].replace(0, pd.NA).last_valid_index()+2*60*6:,:]
        X_train = normalise(train["activity"].values)
        X_test = normalise(test["activity"].values)
        epsilon = 1e-5
        Y_train_soft = train['probability'].clip(lower=epsilon, upper=1 - epsilon)
        Y_test_soft = test['probability'].clip(lower=epsilon, upper=1 - epsilon)
        Y_train_hard = train['majority_vote']
        Y_test_hard = test['majority_vote']
        Y_train_superset = train[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]]
        Y_test_superset = test[["CK", "Sadeh", "HMM", "Sazonova", "Oakley"]]
        update_aggregated_metrics(trad_aggregated, "CK", get_metrics(ground_truth, test["CK"], "CK", window_size))
        update_aggregated_metrics(trad_aggregated, "HMM", get_metrics(ground_truth, test["HMM"], "HMM", window_size))
        update_aggregated_metrics(trad_aggregated, "Sadeh", get_metrics(ground_truth, test["Sadeh"], "Sadeh", window_size))
        update_aggregated_metrics(trad_aggregated, "Sazonova", get_metrics(ground_truth, test["Sazonova"], "Sazonova", window_size))
        update_aggregated_metrics(trad_aggregated, "Oakley", get_metrics(ground_truth, test["Oakley"], "Oakley", window_size))
        update_aggregated_metrics(trad_aggregated, "Ensemble", get_metrics(ground_truth, Y_test_hard, "Ensemble", window_size))
        print("______________________________ENSEMBLE______________________________")
        print(get_metrics(ground_truth, Y_test_hard, "Ensemble", window_size))
        X_train, Y_train_hard, Y_train_soft, Y_train_superset = transform_to_xy_tensor(X_train, Y_train_hard.values, Y_train_soft.values, Y_train_superset.values, window_size)
        X_test, Y_test_hard, Y_test_soft, Y_test_superset = transform_to_xy_tensor(X_test, Y_test_hard.values, Y_test_soft.values, Y_test_superset.values, window_size)
        experiments = {
            "CrossEntropy": {
                "Hard": {
                    "loss_function": nn.CrossEntropyLoss(),
                    "train": Y_train_hard,
                    "test": Y_test_hard
                },
                "Soft": {
                    "loss_function": SoftCrossEntropy(gamma=1),
                    "train": Y_train_soft,
                    "test": Y_test_soft
                }
            },
            "BS": {
                "hard_labels": {
                    "loss_function": BrierScoreLoss(),
                    "train": Y_train_hard,
                    "test": Y_test_hard
                },
                "soft_labels": {
                    "loss_function": SupersetBrierScoreLoss("pessimistic"),
                    "train": Y_train_superset,
                    "test": Y_test_superset
                }
            }
        }
        samples = 50
        results = {}
        for experiment_name, experiment in experiments.items():
            print(f"________________________________{experiment_name}________________________________")
            for type_name, type in experiment.items():
                ref_name = f"{experiment_name}_{type_name}"
                results[ref_name] = []
                print(f"________________________________{type_name}________________________________")
                for name, model in models.items():
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, weight_decay=0.0001)
                    criterion = type["loss_function"]

                    run_training(X_train, type["train"], name, model, criterion, optimizer, epochs=200, batch_size=512, patience=20, min_delta=0.001)
                    metrics, _ = run_testing(X_test, type["test"], ground_truth, name, model, samples)
                    results[ref_name] = metrics
                    print("______________________________NN______________________________")
                    print(metrics)
                    update_aggregated_metrics(aggregated_metrics, ref_name, metrics)
        if (i + 1) % 10 == 0:
            metrics_intermediate = calculate_mean_metrics(aggregated_metrics)
            write_dict_to_json(metrics_intermediate, path_to_save_res / "intermediate_results.json")
            trad_metrics_intermediate = calculate_mean_metrics(trad_aggregated)
            write_dict_to_json(trad_metrics_intermediate, path_to_save_res / "trad_intermediate_results.json")

    final_metrics = calculate_mean_metrics(aggregated_metrics)
    write_dict_to_json(final_metrics, path_to_save_res / "final_results.json")
    trad_metrics_final = calculate_mean_metrics(trad_aggregated)
    write_dict_to_json(trad_metrics_final, path_to_save_res / "trad_final_results.json")


if __name__ == "__main__":
    main()