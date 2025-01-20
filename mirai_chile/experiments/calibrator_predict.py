import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mirai_chile.data.logit import LogitDataset
from mirai_chile.model_evaluation.evaluation_functions.yearly_roc_auc import YearlyROCAUCFunction
from mirai_chile.model_evaluation.evaluation_pipeline import EvaluationPipeline

with open(os.path.expanduser("~/.mirai/snapshots/calibrators/Mirai_calibrator_mar12_2022.p"), 'rb') as infi:
    calibrator = pickle.load(infi)

df = LogitDataset(
    logit_table_path="~/Documents/MirAi/MiraiChile/mirai_chile/data/dataset/combined_logits.csv",
    outcomes_table_path="~/Documents/MirAi/MiraiChile/mirai_chile/data/dataset/outcomes.csv"
).dataframe

df = df.dropna().reset_index(drop=True)

prob_table = []
for idx in range(len(df)):
    row = df.iloc[idx]

    logit = [row[col] for col in df.columns if "logit" in col]
    probs = F.sigmoid(torch.tensor(logit)).data.numpy()
    pred_y = np.zeros(5)

    for i in calibrator.keys():
        pred_y[i] = calibrator[i].predict_proba(probs[i].reshape(-1, 1)).flatten()[1]

    prob_table.append({"identifier": row["identifier"], **{f"year_{i + 1}": pred_y[i] for i in range(5)}})

prob_df = pd.DataFrame(prob_table)
prob_df["time_to_event"] = df["time_to_event"]
prob_df["cancer"] = df["cancer"]
prob_df["machine_manufacturer"] = df["machine_manufacturer"]

print(prob_df.head())

eval_pipe = EvaluationPipeline()
eval_pipe.add_metric(YearlyROCAUCFunction(), {"max_followup": 5})
table = []
for j, m in enumerate(prob_df.machine_manufacturer.unique()):
    subdf = prob_df[prob_df.machine_manufacturer == m]
    eval_pipe.eval_dataset(subdf)
    table.append({"machine_manufacturer": m, **{f"roc_auc_year{i + 1}": eval_pipe.results[j][i] for i in range(5)}})

print(pd.DataFrame(table))
