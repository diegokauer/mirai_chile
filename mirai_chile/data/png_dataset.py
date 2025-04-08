import torch

from mirai_chile.data.abstract_dataset import AbstractDataset
from mirai_chile.data.pre_processing import pre_process_images


class PNGRawDataset(AbstractDataset):
    def __init__(self,
                 dataframe=None,
                 png_table_path="./dataset/png_dataset.csv",
                 outcomes_table_path="./dataset/outcomes.csv",
                 nrows=None,
                 args=None
                 ):
        super().__init__(
            dataframe=dataframe,
            hidden_table_path=png_table_path,
            outcomes_table_path=outcomes_table_path,
            nrows=nrows,
            col_identifier="path"
        )
        self.args = args

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]

        images = [row[col] for col in self.dataframe.columns if self.col_identifier in col]

        try:
            images, batch = pre_process_images(images, self.args)
        except Exception as e:
            print(f"Skipping index {item} due to processing error: {e}")
            raise RuntimeError(f"Index {item} failed")  # PyTorch will skip this sample automatically.

        # Example: Return the images as a dictionary
        return {
            "data": images,
            "time_to_event": torch.tensor(row["time_to_event"]),
            "cancer": torch.tensor(row["cancer"]).long(),
            "device": torch.tensor(row[[col for col in self.dataframe.columns if 'mm_' in col]], dtype=torch.float32),
            "machine_manufacturer": row["machine_manufacturer"],
            "batch": batch,
            "identifier": row["identifier"]}
