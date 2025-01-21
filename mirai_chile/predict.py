import os

import pandas as pd
import torch


def predict_probas(model, dataset, device, dataloader, rank=None, dry_run=False):
    assert dataset in ["logit", "transformer_hidden", "encoder_hidden"]

    model.eval()
    probs_table = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):

            data[dataset] = data[dataset].to(device)

            if "batch" in data:
                for key, val in data["batch"].items():
                    data["batch"][key] = val.to(device)
            else:
                data["batch"] = None

            logit, _, _ = model(data[dataset], data["batch"])
            pmf, s = model.head.logit_to_cancer_prob(logit)

            s_inv = 1 - s

            identifier = data["identifier"]
            for i, id in enumerate(identifier):
                outcomes_dict = {
                    "time_to_event": data["time_to_event"][i].item(),
                    "cancer": data["cancer"][i].item(),
                    "machine_manufacturer": data["machine_manufacturer"][i]
                }
                year_prob_dict = {f"year_{j + 1}": s_inv[i, j].item() for j in range(s_inv.size(1))}
                outcomes_dict.update(year_prob_dict)
                probs_table.append(
                    {"identifier": id, **outcomes_dict})

            if dry_run:
                break

    filename = f"predicted_probas_{dataset}.csv"
    if not rank is None:
        filename = f"predicted_probas_{dataset}_rank_{rank}.csv"

    df = pd.DataFrame(probs_table)
    df.to_csv(os.path.join("./mirai_chile/data/output/", filename), index=False)
    return df
