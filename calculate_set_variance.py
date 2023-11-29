import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch.nn as nn
import seaborn as sns

def calculate_average_activity(activity, timestamps):
    datetimes = pd.to_datetime(timestamps)
    activity = pd.DataFrame(activity, index=datetimes)
    activity_averages = activity.resample('30s').mean()
    return activity_averages.groupby(activity_averages.index.time).mean()

def calculate_averaged_day(predictions, timestamps):
    # Convert timestamps to datetime objects
    datetimes = pd.to_datetime(timestamps)
    predictions.index = datetimes
    daily_averages = predictions.resample('30S').mean()
    # Now group by time to get mean and confidence intervals across all days
    grouped = daily_averages.groupby(daily_averages.index.time)
    daily = grouped.mean()
    return daily

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
    columns_to_switch = ["CK", "Sadeh", "HMM", "Sazonova", "Oakley", "wake"]
    df = switch_ones_and_zeros(df, columns_to_switch)
    df["majority_vote"] = majority_vote(df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]])
    df["probability"] = df[["CK",  "Sadeh", "HMM",  "Sazonova",  "Oakley"]].mean(axis=1)
    return df


def get_acc_files(path_to_acc_files):
    return os.listdir(path_to_acc_files)

def get_file_names(insomnia, non_insomnia, file_list, variable):
    variable_files = []
    non_variable_files = []
    for file in file_list:
        file_number = int(file.split('_')[1].split('.')[0])
        if file_number in insomnia:
            variable_files.append(file)
        elif file_number in non_insomnia:
            non_variable_files.append(file)
            
    return {f"{variable}": variable_files, f"non-{variable}": non_variable_files}


def get_type_files(mesa_df: pd.DataFrame, file_names, variable_name):
    non_apnea = mesa_df.loc[mesa_df["slpapnea5"]==0, :]
    variable = non_apnea.loc[non_apnea[variable_name]==1, "mesaid"].to_list()
    non_variable = non_apnea.loc[non_apnea[variable_name]==0, "mesaid"].sample(n=len(variable)).to_list()
    return get_file_names(variable, non_variable, file_names, variable_name)


def main():
    path_to_data = Path(__file__).parent / "data"
    path_to_files =  path_to_data / "ensemble_files"
    path_to_mesa_file = Path(__file__).parent / "data" / "mesa-sleep-dataset-0.6.0.csv"
    mesa_data = pd.read_csv(path_to_mesa_file, engine="python")
    window_size = 20
    acc_file_names = get_acc_files(path_to_files)
    files = get_type_files(mesa_data, acc_file_names, "insmnia5")
    for type, file_names in files.items():
        mean_list = []
        for i, file_name in enumerate(file_names):
            print(f"Processing: {file_name} Nr. {i} of {len(file_names)}")
            if file_name in ["Acc_2105.csv"]:
                continue
            acc_file = pd.read_csv(path_to_files / file_name, engine="python")
            df = create_dataset(acc_file)
            set_variance = df[['CK', 'Sadeh', 'HMM', 'Sazonova', 'Oakley',]].var(axis=1)
            mean_var = calculate_averaged_day(set_variance, df['linetime.1'])
            mean_list.append(mean_var)
        variance_df = pd.concat(mean_list, axis=1)
        variance_df.to_csv(path_to_data/ f"{type}_set_variance.csv")

if __name__ == "__main__":
    main()