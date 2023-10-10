import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np


def find_file(files, participant_id):
    return [file for file in files if str(participant_id) in file]


def read_in_join_table(path_to_join_table):
    return pd.read_csv(path_to_join_table, engine="python")


def add_ground_truth(acc_file: pd.DataFrame, psg_starting_line, psg):
    acc_file["sleep_stages"] = [0]*acc_file.shape[1]

    return


def extract_sleep_stages(psg_xml):
    events = []
    # Iterate through the XML to get the desired events
    for scored_event in psg_xml.findall('./ScoredEvents/ScoredEvent'):
        event_type = scored_event.find('EventType').text

        if event_type == "Stages|Stages":
            start_time = float(scored_event.find('Start').text)
            duration = float(scored_event.find('Duration').text)
            concept = scored_event.find('EventConcept').text.split('|')[-1]

            # Append the data to the list for the duration of the event
            for i in range(int(duration)):
                events.append((int(start_time) + i, concept))

    # Convert the list of events to a pandas DataFrame
    df = pd.DataFrame(events, columns=['Second', 'EventConcept'])
    df['EventConcept'] = df['EventConcept'].astype(int)
    aggregated_series = df.groupby(df['Second'] // 30)['EventConcept'].agg(lambda x: x.mode().iloc[0])
    return aggregated_series


def read_csv_file(path: str, engine: str = "python") -> pd.DataFrame:
    return pd.read_csv(path, engine=engine)


def adjust_linetime(df: pd.DataFrame) -> pd.DataFrame:
    first_line = df['linetime'].iloc[0]
    timestamp_str = f'01.01.85 {first_line}'
    starting_timestamp = pd.Timestamp(timestamp_str)

    seconds_series = np.arange(0, len(df) * 30, 30)
    df['linetime'] = starting_timestamp + pd.to_timedelta(seconds_series, unit='s')
    return df


def adjust_sleep_stage_index(sleep_stages: pd.Series, starting_time: str) -> pd.Series:
    sleep_stages.index = [
        pd.Timestamp(f'01.01.85 {starting_time}') + pd.Timedelta(seconds=interval * 30)
        for interval in sleep_stages.index
    ]
    return sleep_stages


def read_in_participant(participant_keys, path_to_acc_files, path_to_psg_files) -> pd.DataFrame:
    participant_id = "{:04}".format(participant_keys["mesaid"])

    acc_file = find_file(os.listdir(path_to_acc_files), participant_id)
    if not acc_file:
        return pd.DataFrame([])
    acc_df = read_csv_file(path_to_acc_files / acc_file[0])
    acc_df = adjust_linetime(acc_df)

    psg_file = find_file(os.listdir(path_to_psg_files), participant_id)
    if not psg_file:
        return pd.DataFrame([])
    sleep_stages_xml_root = ET.parse(path_to_psg_files / psg_file[0])
    sleep_stages = extract_sleep_stages(sleep_stages_xml_root)

    sleep_stages = adjust_sleep_stage_index(sleep_stages, participant_keys["linetime"])
    sleep_stages = sleep_stages.reset_index()
    sleep_stages.columns = ['linetime', 'ground_truth']

    joint_table = pd.merge(acc_df, sleep_stages, on='linetime', how='left')
    joint_table['ground_truth'].fillna(0, inplace=True)
    joint_table.index = joint_table['linetime']

    return joint_table


def main():
    path_to_acc_files = Path(__file__).parent / "data" / "mesa" / "actigraphy"
    path_to_psg_files = Path(__file__).parent / "data" / "mesa" / "polysomnography" / "annotations-events-nsrr"
    path_to_overlap_file = Path(__file__).parent / "data" / "mesa" / "overlap" / "mesa-actigraphy-psg-overlap.csv"
    path_to_save_files = Path(__file__).parent / "data" / "ground_truth_activity"
    join_table = read_in_join_table(path_to_overlap_file)
    for index, row in join_table.iterrows():
        print(f'Current ID: {row["mesaid"]}')
        joint_table = read_in_participant(row, path_to_acc_files, path_to_psg_files)
        if joint_table.empty:
            continue
        joint_table.to_csv(path_to_save_files / f'Acc_{row["mesaid"]}.csv')
        print(f"Processed {index} out of {len(join_table)}")


if __name__ == "__main__":
    main()