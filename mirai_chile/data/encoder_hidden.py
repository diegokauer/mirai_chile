import pandas as pd
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
        row = self.dataframe.iloc[idx]

        encoder_hidden = [row[col] for col in self.dataframe.columns if "hidden" in col]

        return {
            "encoder_hidden": encoder_hidden,
            "time_to_event": row["time_to_event"],
            "cancer": row["cancer"],
            "machine_manufacturer": row["machine_manufacturer"],
            "identifier": row["identifier"]
        }

    def generate_encoder_hidden_dataframe(self):
        encoder_hidden_df = pd.read_csv(self.encoder_hidden_table_path, nrows=self.nrows)
        outcome_df = pd.read_csv(self.outcomes_table_path)

        encoder_hidden_df = encoder_hidden_df.merge(outcome_df)

        return encoder_hidden_df.dropna()

    def get_split(self, split):
        return EncoderHiddenDataset(self.dataframe[self.dataframe.split == split])
