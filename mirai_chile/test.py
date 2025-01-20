import pandas as pd
import torch


def test_model(model, dataset, device, dataloader, eval_pipeline=None, dry_run=False):
    assert dataset in ["logit", "transformer_hidden", "encoder_hidden"]

    test_loss = 0
    probs_table = []
    model.eval()

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
            t = data["time_to_event"].to(device)
            d = data["cancer"].to(device)

            test_loss += model.loss_function(logit, pmf, s, t, d).item()

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

    df = pd.DataFrame(probs_table)

    if not eval_pipeline is None:
        eval_pipeline.eval_dataset(df)
        print(eval_pipeline)
