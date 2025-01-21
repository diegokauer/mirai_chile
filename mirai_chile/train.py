from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


def train_model(model, dataset, device, dataloader, optimizer, epoch, dry_run=False):
    assert dataset in ["logit", "transformer_hidden", "encoder_hidden"]

    train_loss = 0

    model.train()

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1} - Train", unit="batch") as pbar:
        for batch_idx, data in enumerate(dataloader):

            data[dataset] = data[dataset].to(device)

            if "batch" in data:
                for key, val in data["batch"].items():
                    data["batch"][key] = val.to(device)
            else:
                data["batch"] = None

            optimizer.zero_grad()
            logit, _, _ = model(data[dataset], data["batch"])
            # print(logit)
            if isinstance(model, DDP):
                pmf, s = model.module.head.logit_to_cancer_prob(logit)
            else:
                pmf, s = model.head.logit_to_cancer_prob(logit)
            t = data["time_to_event"].to(device)
            d = data["cancer"].to(device)

            loss = model.loss_function(logit, pmf, s, t, d)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            current_train_loss = train_loss / (pbar.n + 1)  # Average loss up to the current batch
            pbar.set_postfix(
                {'Batch Loss': f"{loss.item():.4f}", 'Current Train Loss': f"{current_train_loss:.4f}"})
            pbar.update(1)

    print('\nTrain set: Total loss: {:.6f}\tAverage loss: {:.6f}\tAverage batch loss: {:.6f}'.format(
        train_loss, train_loss / len(dataloader.dataset), train_loss / len(dataloader)))
