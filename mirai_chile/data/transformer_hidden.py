import pandas as pd
import torch
from torch.utils.data import Dataset


class TransformerHiddenDataset(Dataset):
    def __init__(self,
                 dataframe=None,
                 transformer_hidden_table_path="mirai_chile/data/dataset/combined_transformer_hidden.csv",
                 outcomes_table_path="mirai_chile/data/dataset/outcomes.csv",
                 nrows=None):
        super().__init__()
        self.dataframe = dataframe
        self.transformer_hidden_table_path = transformer_hidden_table_path
        self.outcomes_table_path = outcomes_table_path
        self.nrows = nrows
        if self.dataframe is None:
            self.dataframe = self.generate_transformer_hidden_dataframe()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]

        transformer_hidden = [row[col] for col in self.dataframe.columns if "hidden" in col]

        return {
            "transformer_hidden": torch.tensor(transformer_hidden, dtype=torch.float32),
            "time_to_event": torch.tensor(row["time_to_event"]),
            "cancer": torch.tensor(row["cancer"]).long(),
            "machine_manufacturer": row["machine_manufacturer"],
            "identifier": row["identifier"]
        }

    def generate_transformer_hidden_dataframe(self):
        transformer_df = pd.read_csv(self.transformer_hidden_table_path, nrows=self.nrows)
        outcome_df = pd.read_csv(self.outcomes_table_path)

        transformer_df = transformer_df.merge(outcome_df)

        return transformer_df.dropna()

    def get_split(self, split):
        return TransformerHiddenDataset(self.dataframe[self.dataframe.split == split])

    def get_manufacturer(self, manufacturer):
        return TransformerHiddenDataset(self.dataframe[self.dataframe.machine_manudacturer == manufacturer])
