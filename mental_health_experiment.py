import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import torch.nn as nn
import seaborn as sns


from algorithms.bayesian_net import BayesianLSTM
from sklearn.model_selection import train_test_split
from algorithms.uncertainty_measures import ar_covariance_estimation
from algorithms.loss_function import BrierScoreLoss, SoftCrossEntropy, SupersetBrierScoreLoss

def calculate_average_activity(activity, timestamps):
    datetimes = pd.to_datetime(timestamps)
    activity = pd.DataFrame(activity, index=datetimes)
    activity_averages = activity.resample('30s').mean()
    return activity_averages.groupby(activity_averages.index.time).mean()

def calculate_averaged_day(predictions, timestamps):
    # Convert timestamps to datetime objects
    datetimes = pd.to_datetime(timestamps)
    
    # Create a DataFrame to hold the predictions and timestamps
    preds = pd.DataFrame(predictions.numpy().T, index=datetimes)
    # Resample the data to daily averages for each 30-second epoch
    daily_averages = preds.resample('60S').mean()
    # Now group by time to get mean and confidence intervals across all days
    grouped = daily_averages.groupby(daily_averages.index.time)
    daily_mean_predictions = grouped.mean()
    mean_predictions = daily_mean_predictions.mean(axis=1)
    std_predictions = daily_mean_predictions.std(axis=1)
    return mean_predictions.squeeze(), std_predictions.squeeze()

def calculate_averaged_day_ensemble(predictions, timestamps):
    # Convert timestamps to datetime objects
    datetimes = pd.to_datetime(timestamps)
    predictions.index = datetimes
    # Create a DataFrame to hold the predictions and timestamps
    # Resample the data to daily averages for each 30-second epoch
    daily_averages = predictions.resample('30S').mean()
    # Now group by time to get mean and confidence intervals across all days
    grouped = daily_averages.groupby(daily_averages.index.time)
    daily_mean_predictions = grouped.mean()
    mean_predictions = daily_mean_predictions.mean(axis=1)
    return mean_predictions.squeeze(), []


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
    df = switch_ones_and_zeros(df, columns_to_switch)
    df["majority_vote"] = majority_vote(df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]])
    df["probability"] = df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]].mean(axis=1)
    return df


def transform_to_xy_tensor(X, labels, window_size):
    X_data, Y_soft = [], []
    for i in range(len(X) - window_size):
        X_data.append(X[i:i + window_size])
        Y_soft.append(labels[i + window_size])
    X_data = np.array(X_data)
    Y_soft = np.array(Y_soft)
    return torch.tensor(X_data).float().cuda(), torch.tensor(Y_soft).float().cuda()


def train(model, optimizer, criterion, X, Y, epochs, batch_size, patience, min_delta):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    X_train = X_train.unsqueeze(2)
    X_test = X_test.unsqueeze(2)
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # Create DataLoader for test set
    test_data = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size,  drop_last=True)
    best_test_loss = float('inf')  # initialize with a high value
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
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
                inputs, labels = inputs.cuda(), labels.cuda()
                test_outputs = model(inputs)
                if test_outputs.squeeze().dim() == 0:
                    continue
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


def get_type_files(file_names):
    files = {"control": [],"schizophrenia": [], "depression": []}
    for file_name in file_names:
        state = [state_name for state_name in ["control","schizophrenia", "depression"] if state_name in file_name][0]
        files[state].append(file_name)
    return files


def run_testing(X_test, classifier, samples):
    X_test = X_test.unsqueeze(2)
    classifier.train()
    predictions = []
    for _ in range(samples):
        with torch.no_grad():
            outputs = classifier(X_test).cpu()
            predictions.append(outputs.squeeze())
    return torch.stack(predictions)


def main():
    path_to_data = Path(__file__).parent / "data" / "mental_health"
    path_to_files =  path_to_data / "ensemble_files"
    window_size = 10
    acc_file_names = get_acc_files(path_to_files)
    files = get_type_files(acc_file_names)
    samples = 50
    for type, file_names in files.items():
        print(type)
        mean_list = []
        std_list = []
        for i, file_name in enumerate(file_names):
            print(f"Processing: {file_name} Nr. {i} of {len(file_names)}")
            acc_file = pd.read_csv(path_to_files / file_name, engine="python")
            df = create_dataset(acc_file)
            X_ = normalise(df["activity"].values)
            epsilon = 1e-5
            Y_soft =df["majority_vote"]#.clip(lower=epsilon, upper=1 - epsilon)
            X, Y_soft = transform_to_xy_tensor(X_, Y_soft.values, window_size)
            model = BayesianLSTM(input_dim=1, hidden_dim=25, dropout=0.5, num_layers=2, temperature=2).cuda()
            #optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, weight_decay=0.0001)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
            criterion = BrierScoreLoss()
            train(model, optimizer, criterion, X, Y_soft, epochs=200, batch_size=512, patience=20, min_delta=0.001)
            predictions = run_testing(X, model, samples)
            mean_preds, std_preds = calculate_averaged_day(predictions, df.loc[window_size:,'timestamp'])
            mean_list.append(mean_preds)
            std_list.append(std_preds)
        mean_df = pd.concat(mean_list, axis=1)
        std_df = pd.concat(std_list, axis=1)
        mean_df.to_csv(path_to_data/ f"{type}_brier_score_adam_mean_predictions_LSTM.csv")
        std_df.to_csv(path_to_data/ f"{type}_brier_score_adam_std_predictions_LSTM.csv")

if __name__ == "__main__":
    main()
