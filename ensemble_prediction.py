import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from algorithms.traditional_algorithms import calculate_cole_kripke, sadeh_sleep_stage_algo, fourier_based_hmm, sazonova_algorithm, calculate_oakley


def apply_trad_algos(path_to_folder, algorithms, path_to_save_files):
    files = os.listdir(path_to_folder)
    acc_dfs = []
    for file in files:
        acc_file = pd.read_csv(path_to_folder / file, engine="python")
        acc_file = acc_file.dropna()
        for algorithm_name, algorithm in algorithms.items():
            acc_file[algorithm_name] = algorithm(acc_file["activity"], 30)
        acc_file.to_csv(path_to_save_files / file)
        print(f"{file} processed")


def main():
    path_to_acc_files = Path(__file__).parent / "data" / "ground_truth_activity"
    path_to_save_files = Path(__file__).parent / "data" / "ensemble_files"
    algorithms = {"CK": calculate_cole_kripke, "Sadeh": sadeh_sleep_stage_algo, "HMM": fourier_based_hmm,
                  "Sazonova": sazonova_algorithm, "Oakley": calculate_oakley}
    apply_trad_algos(path_to_acc_files, algorithms, path_to_save_files)


if __name__ == "__main__":
    main()

