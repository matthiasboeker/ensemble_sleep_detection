import os
from pathlib import Path
import pandas as pd
from typing import Dict
from pathlib import Path
import pandas as pd
import os
from algorithms.traditional_algorithms import calculate_cole_kripke_30sec, calculate_cole_kripke, sadeh_sleep_stage_algo, fourier_based_hmm, sazonova_algorithm, calculate_oakley

def get_index_to_align(activity_df: pd.DataFrame):
    times = activity_df["timestamp"].apply(lambda x: x.split()[1])
    first_index = times.loc[times == "12:00:00"].index[0]
    return first_index

def read_in_files_in_folder(path_to_folder: Path, algorithms, condition: str, path_to_save_folder: Path) -> Dict[str, pd.DataFrame]:
    files = []
    file_names = os.listdir(path_to_folder)
    for nr, file_path in enumerate(file_names):
        file = pd.read_csv(path_to_folder / file_path, delimiter=None, engine="python")
        f_index = get_index_to_align(file)
        activity_file = file.iloc[f_index:, :]
        activity_file.index = pd.DatetimeIndex(activity_file["timestamp"])
        activity_file = activity_file.drop(["date", "timestamp"], axis=1)
        for algorithm_name, algorithm in algorithms.items():
            activity_file[algorithm_name] = algorithm(activity_file["activity"], 60)
        activity_file.to_csv(path_to_save_folder / f"mental_health_acc_{nr}_{condition}.csv")
        print(f"Create file {nr} of {len(file_names)}")
    return files

def main():
    path_to_data = Path(__file__).parent / "data" / "mental_health"
    path_to_schizophrenia = path_to_data / "schizophrenia"
    path_to_depression = path_to_data / "depression"                     
    path_to_control = path_to_data / "control"     
    path_to_store = path_to_data / "ensemble_files"                                                                   
    algorithms = {"CK": calculate_cole_kripke, "Sadeh": sadeh_sleep_stage_algo, "HMM": fourier_based_hmm,
                  "Sazonova": sazonova_algorithm, "Oakley": calculate_oakley}
    read_in_files_in_folder(path_to_schizophrenia, algorithms, "schizophrenia" ,path_to_store)
    print("Done schizophrenia")
    read_in_files_in_folder(path_to_depression, algorithms, "depression" ,path_to_store)
    print("Done depression")
    read_in_files_in_folder(path_to_control, algorithms, "control" ,path_to_store)
    print("Done control")

if __name__ == "__main__":
    main()

