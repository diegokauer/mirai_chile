import pandas as pd
import torch
from torch.utils.data import Dataset


class LogitDataset(Dataset):
    def __init__(self,
                 dataframe=None,
                 logit_table_path="./dataset/combined_logits.csv",
                 outcomes_table_path="./dataset/outcomes.csv",
                 nrows=None):
        super().__init__()
        self.dataframe = dataframe
        self.logit_table_path = logit_table_path
        self.outcomes_table_path = outcomes_table_path
        self.nrows = nrows
        if self.dataframe is None:
            self.dataframe = self.generate_logit_dataframe()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]

        logit = [row[col] for col in self.dataframe.columns if "logit" in col]

        return {
            "logit": torch.tensor(logit, dtype=torch.float32),
            "time_to_event": torch.tensor(row["time_to_event"]),
            "cancer": torch.tensor(row["cancer"]).long(),
            "machine_manufacturer": row["machine_manufacturer"],
            "identifier": row["identifier"]
        }

    def generate_logit_dataframe(self):
        logit_df = pd.read_csv(self.logit_table_path, nrows=self.nrows)
        outcome_df = pd.read_csv(self.outcomes_table_path)

        logit_df = logit_df.merge(outcome_df)

        return logit_df.dropna()

    def get_split(self, split):
        return LogitDataset(self.dataframe[self.dataframe.split == split])
