import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def test_model(model, loss_function, dataset, device, dataloader, eval_pipeline=None, dry_run=False,
               print_result=False):
    assert dataset in ["logit", "transformer_hidden", "encoder_hidden"]

    test_loss = 0
    probs_table = []
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):

            data['data'] = data['data'].to(device)

            if "batch" in data:
                for key, val in data["batch"].items():
                    data["batch"][key] = val.to(device)
            else:
                data["batch"] = None

            logit, _, _ = model(data["data"], data["batch"])
            if isinstance(model, DDP):
                pmf, s = model.module.head.logit_to_cancer_prob(logit)
            else:
                pmf, s = model.head.logit_to_cancer_prob(logit)
            data["time_to_event"] = data["time_to_event"].to(device)
            data["cancer"] = data["cancer"].to(device)
            data["logit"] = logit.to(device)
            data["pmf"] = pmf.to(device)

            loss = loss_function(data)
            test_loss += loss.item()

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

    print('Test set: Total loss: {:.6f}\tAverage loss: {:.6f}\tAverage batch loss: {:.6f}\n'.format(
        test_loss, test_loss / len(dataloader.dataset), test_loss / len(dataloader)))

    data = pd.DataFrame(probs_table)
    print(data[data.cancer == 1].head())
    print(data[data.cancer == 0].head())

    if not eval_pipeline is None:
        eval_pipeline.eval_dataset(data)
        if print_result:
            print(eval_pipeline)

    return eval_pipeline
