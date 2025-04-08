import pandas as pd
import torch
from torch.utils.data import Dataset

from mirai_chile.data.utils.train_statistics import get_censoring_dist


class AbstractDataset(Dataset):
    def __init__(self,
                 dataframe=None,
                 hidden_table_path=None,
                 outcomes_table_path="./dataset/outcomes.csv",
                 nrows=None,
                 censoring_dist=None,
                 col_identifier=""):
        super().__init__()
        self.dataframe = dataframe
        self.outcomes_table_path = outcomes_table_path
        self.hidden_table_path = hidden_table_path
        self.nrows = nrows
        self.censoring_dist = censoring_dist
        self.col_identifier = col_identifier
        if self.dataframe is None:
            self.dataframe = self.generate_hidden_dataframe()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]

        hidden = [row[col] for col in self.dataframe.columns if self.col_identifier in col]

        batch = {
            "time_seq": torch.tensor([0, 0, 0, 0]),
            "view_seq": torch.tensor([0, 0, 1, 1]),
            "side_seq": torch.tensor([1, 0, 1, 0])
        }

        return {
            "data": torch.tensor(hidden, dtype=torch.float32),
            "time_to_event": torch.tensor(row["time_to_event"]),
            "cancer": torch.tensor(row["cancer"]).long(),
            "device": torch.tensor(row[[col for col in self.dataframe.columns if 'mm_' in col]], dtype=torch.float32),
            "machine_manufacturer": row["machine_manufacturer"],
            "batch": batch,
            "identifier": row["identifier"],
        }

    def generate_hidden_dataframe(self):
        hidden_df = pd.read_csv(self.hidden_table_path, nrows=self.nrows)
        outcome_df = pd.read_csv(self.outcomes_table_path)

        hidden_df = hidden_df.merge(outcome_df)
        machine_manufacturer = hidden_df.machine_manufacturer
        hidden_df = pd.get_dummies(hidden_df, columns=['machine_manufacturer'], prefix='mm_')
        hidden_df['machine_manufacturer'] = machine_manufacturer
        hidden_df.dropna(inplace=True)

        self.censoring_dist = get_censoring_dist(hidden_df[(hidden_df.split == 'train') & (~hidden_df.cancer)])

        return hidden_df

    def get_split(self, split):
        return self.__class__(self.dataframe[self.dataframe.split == split],
                              censoring_dist=self.censoring_dist)

    def get_splits(self, splits):
        return self.__class__(self.dataframe[self.dataframe.split.isin(splits)],
                              censoring_dist=self.censoring_dist)

    def get_manufacturer(self, manufacturer):
        return self.__class__(self.dataframe[self.dataframe.machine_manufacturer == manufacturer],
                              censoring_dist=self.censoring_dist)
