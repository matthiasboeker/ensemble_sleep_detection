import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import seaborn as sns


from algorithms.bayesian_net import BayesianCNN, BayesianNet
from sklearn.model_selection import train_test_split
from algorithms.uncertainty_measures import ar_covariance_estimation
from algorithms.loss_function import SoftCrossEntropy


def calculate_averaged_day(predictions, timestamps):
    # Convert timestamps to datetime objects
    datetimes = pd.to_datetime(timestamps)
    
    # Create a DataFrame to hold the predictions and timestamps
    df = pd.DataFrame(predictions.numpy().T, index=datetimes)
    
    # Shift time so the day starts at 9 am
    df.index = df.index.shift(-9, freq='H')
    
    # Resample the data to daily averages for each 30-second epoch
    daily_averages = df.resample('30S').mean()
    
    # Now group by time to get mean and confidence intervals across all days
    grouped = daily_averages.groupby(daily_averages.index.time)
    daily_mean_predictions = grouped.mean()
    mean_predictions = daily_mean_predictions.mean(axis=1)
    lower_bound = daily_mean_predictions.quantile(0.25, axis=1)
    upper_bound = daily_mean_predictions.quantile(0.75, axis=1)
    return mean_predictions.squeeze(), lower_bound.squeeze(), upper_bound.squeeze()


def plot_predictive_interval(predictions, timestamps, path_to_save_file, name, title='95% Predictive Interval for an Averaged Day'):
    # Get averaged day statistics
    mean_predictions, lower_bound, upper_bound = calculate_averaged_day(predictions, timestamps)
    time_vector = pd.to_datetime(mean_predictions.index, format='%H:%M:%S')
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Plot mean predictions and confidence interval
    ax.plot(time_vector, mean_predictions, label='Mean Prediction')
    ax.fill_between(time_vector, lower_bound, upper_bound, alpha=0.2, label='95% Predictive Interval')
    ax.axhline(0.5, c="r")
    # Optional: Customize the plot
    ax.set_title(title)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Prediction Value')
    ax.legend()
    
    # Show plot
    plt.savefig(path_to_save_file / f"predictive_interval_averaged_day_{name}.png")


def plot_predictive_distribution(predictions, bins=50, title='Predictive Distribution with Monte Carlo Dropout'):
    # Convert tensor to numpy array
    predictions_np = predictions.numpy()

    # Flatten the predictions to get a 1D array of all prediction values
    flattened_predictions = predictions_np.flatten()

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(flattened_predictions, bins=bins, density=True, alpha=0.75, edgecolor='black', label='Predictive Distribution')

    # Plot KDE
    sns.kdeplot(flattened_predictions, ax=ax, color='red', label='KDE')

    # Optional: Customize the plot
    ax.set_title(title)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Density')

    # Show plot
    plt.show()

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
    columns_to_switch = ["CK", "Sadeh", "HMM", "Sazonova", "Oakley", "wake"]
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
    return torch.tensor(X_data).float(), torch.tensor(Y_soft).float()


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

def get_file_names(insomnia, non_insomnia, file_list):
    insomnia_files = []
    non_insomnia_files = []
    for file in file_list:
        file_number = int(file.split('_')[1].split('.')[0])
        if file_number in insomnia:
            insomnia_files.append(file)
        elif file_number in non_insomnia:
            non_insomnia_files.append(file)
            
    return {"non-insomnia": non_insomnia_files, "insomnia": insomnia_files}


def get_insomnia_files(mesa_df: pd.DataFrame, file_names):
    non_apnea = mesa_df.loc[mesa_df["slpapnea5"]==0, :]
    insomnia = non_apnea.loc[non_apnea["insmnia5"]==1, "mesaid"].to_list()
    non_insomnia = non_apnea.loc[non_apnea["insmnia5"]==0, "mesaid"].sample(n=len(insomnia)).to_list()
    return get_file_names(insomnia, non_insomnia, file_names)


def run_testing(X_test, classifier, samples):
    classifier.train()
    predictions = []
    for _ in range(samples):
        with torch.no_grad():
            outputs = classifier(X_test)
            predictions.append(outputs.squeeze())
    return torch.stack(predictions)


def main():
    path_to_data = Path(__file__).parent / "data" / "ensemble_files"
    path_to_mesa_file = Path(__file__).parent / "data" / "mesa-sleep-dataset-0.6.0.csv"
    path_to_figures = Path(__file__).parent / "figures"
    mesa_data = pd.read_csv(path_to_mesa_file, engine="python")
    window_size = 20
    acc_file_names = get_acc_files(path_to_data)
    files = get_insomnia_files(mesa_data, acc_file_names)
    samples = 50
    for type, file_names in files.items():
        print(type)
        for i, file_name in enumerate(file_names):
            print(f"Processing: {file_name} Nr. {i}")
            acc_file = pd.read_csv(path_to_data / file_name, engine="python")
            df = create_dataset(acc_file)
            X = normalise(df["activity"].values)
            epsilon = 1e-5
            Y_soft = df['probability'].clip(lower=epsilon, upper=1 - epsilon)
            X, Y_soft = transform_to_xy_tensor(X, Y_soft.values, window_size)
            model = BayesianCNN(window_size=window_size, dropout=0.5, temperature=2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, weight_decay=0.0001)
            criterion = SoftCrossEntropy(gamma=1)
            train(model, optimizer, criterion, X, Y_soft, epochs=200, batch_size=512, patience=20, min_delta=0.001)
            predictions = run_testing(X, model, samples)
            plot_predictive_interval(predictions, df.loc[window_size:,"linetime.1"], path_to_figures, file_name)
            variance = ar_covariance_estimation(predictions)**0.5
            


if __name__ == "__main__":
    main()