from mirai_chile.data.abstract_dataset import AbstractDataset


class TransformerHiddenDataset(AbstractDataset):
    def __init__(self,
                 dataframe=None,
                 transformer_hidden_table_path="mirai_chile/data/dataset/combined_transformer_hidden.csv",
                 outcomes_table_path="mirai_chile/data/dataset/outcomes.csv",
                 nrows=None):
        super().__init__(
            dataframe=dataframe,
            hidden_table_path=transformer_hidden_table_path,
            outcomes_table_path=outcomes_table_path,
            nrows=nrows,
            col_identifier="hidden"
        )
