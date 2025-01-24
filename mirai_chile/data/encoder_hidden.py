import pandas as pd
import torch
from torch.utils.data import Dataset


class EncoderHiddenDataset(Dataset):
    def __init__(self,
                 dataframe=None,
                 encoder_hidden_table_path="./dataset/combined_encoder_hidden.csv",
                 outcomes_table_path="./dataset/outcomes.csv",
                 nrows=None):
        super().__init__()
        self.dataframe = dataframe
        self.encoder_hidden_table_path = encoder_hidden_table_path
        self.outcomes_table_path = outcomes_table_path
        self.nrows = nrows
        if self.dataframe is None:
            self.dataframe = self.generate_encoder_hidden_dataframe()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]

        encoder_hidden = [row[col] for col in self.dataframe.columns if "encoder" in col]

        batch = {
            "time_seq": torch.tensor([0, 0, 0, 0]),
            "view_seq": torch.tensor([0, 0, 1, 1]),
            "side_seq": torch.tensor([1, 0, 1, 0])
        }

        return {
            "encoder_hidden": torch.tensor(encoder_hidden, dtype=torch.float32),
            "time_to_event": torch.tensor(row["time_to_event"]),
            "cancer": torch.tensor(row["cancer"]).long(),
            "machine_manufacturer": row["machine_manufacturer"],
            "batch": batch,
            "identifier": row["identifier"]
        }

    def generate_encoder_hidden_dataframe(self):
        encoder_hidden_df = pd.read_csv(self.encoder_hidden_table_path, nrows=self.nrows)
        outcome_df = pd.read_csv(self.outcomes_table_path)

        encoder_hidden_df = encoder_hidden_df.merge(outcome_df)

        return encoder_hidden_df.dropna()

    def get_split(self, split):
        return EncoderHiddenDataset(self.dataframe[self.dataframe.split == split])

    def get_manufacturer(self, manufacturer):
        return EncoderHiddenDataset(self.dataframe[self.dataframe.machine_manudacturer == manufacturer])
