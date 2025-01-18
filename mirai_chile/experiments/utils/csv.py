import pandas as pd
import os
from glob import glob


def combine_csv(directory, file_pattern):
    directory = os.path.expanduser(directory)
    csv_files = glob.glob(os.path.join(directory, f"{file_pattern}_rank_*.csv"))

    dataframes = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.to_csv(os.path.join(directory, f"combined_{file_pattern}.csv"), index=False)
