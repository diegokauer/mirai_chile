from mirai_chile.data.abstract_dataset import AbstractDataset


class EncoderHiddenDataset(AbstractDataset):
    def __init__(self,
                 dataframe=None,
                 encoder_hidden_table_path="./dataset/combined_encoder_hidden.csv",
                 outcomes_table_path="./dataset/outcomes.csv",
                 nrows=None):
        super().__init__(
            dataframe=dataframe,
            hidden_table_path=encoder_hidden_table_path,
            outcomes_table_path=outcomes_table_path,
            nrows=nrows,
            col_identifier="encoder"
        )
