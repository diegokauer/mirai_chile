import pandas as pd

from mirai_chile.model_evaluation.evaluation_functions.abstract_function import AbstractFunction
from mirai_chile.model_evaluation.evaluation_functions.yearly_roc_auc import get_yearly_roc_auc


def roc_auc_manufacturer(df, max_followup):
    table = []
    for m in df.machine_manufacturer.unique():
        subdf = df[df.machine_manufacturer == m]
        probs = get_yearly_roc_auc(subdf, max_followup)
        table.append({"machine_manufacturer": m, **{f"roc_auc_year{i + 1}": probs[i] for i in range(max_followup)}})

    return pd.DataFrame(table)


class YearlyROCAUCManufacturerFunction(AbstractFunction):
    def __init__(self, function=roc_auc_manufacturer):
        super().__init__()
        self.name = "Yearly ROC-AUC by Machine Manufacturer"
        self.function = function
