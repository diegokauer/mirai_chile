import torch


def test_model(model, dataset, device, dataloader, dry_run=False):
    assert dataset in ["logit", "transformer_hidden", "encoder_hidden"]

    test_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):

            data[dataset] = data[dataset].to(device)

            if "batch" in data:
                for key, val in data["batch"].items():
                    data["batch"][key] = val.to(device)

            logit, _, _ = model(data[dataset])
            pmf, s = model.head.logit_to_cancer_prob(logit)
            t = data["time_to_event"].to(device)
            d = data["cancer"].to(device)

            test_loss += model.loss_function(logit, pmf, s, t, d).item()

            if dry_run:
                break

    print('Test set: Total loss: {:.6f}\tAverage loss: {:.6f}\tAverage batch loss: {:.6f}\n'.format(
        test_loss, test_loss / len(dataloader.dataset), test_loss / len(dataloader)))
