import numpy as np
import torch
from tqdm import tqdm

from mirai_chile.test import test_model
from mirai_chile.train_step import mirai_step


def train_epoch(models, optimizers, loss_functions, device, dataloaders, epoch, dry_run=False):

    train_loss = 0
    train_dataloader = dataloaders['train']

    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch} - Train", unit="batch") as pbar:
        for batch_idx, data in enumerate(train_dataloader):

            batch_loss, _ = mirai_step(data, models, optimizers, device, loss_functions, dry_run)

            batch_loss = batch_loss.item()
            train_loss += batch_loss
            current_train_loss = train_loss / (pbar.n + 1)  # Average loss up to the current batch

            pbar.set_postfix(
                {'Batch Loss': f"{batch_loss:.4f}", 'Current Train Loss': f"{current_train_loss:.4f}"})
            pbar.update(1)

            if dry_run:
                break

    return current_train_loss


def train_model(models, loss_functions, dataset, device, dataloaders, optimizers, epochs, eval_pipeline, dry_run=False):
    assert dataset in ["logit", "transformer_hidden", "encoder_hidden"]

    patience = 5
    best_metric = -np.inf
    best_state = None
    counter = 0

    models['mirai'].train()
    models['discriminator'].train()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(models, optimizers, loss_functions, device, dataloaders, epoch, dry_run)

        results = test_model(
            models['mirai'],
            loss_functions['mirai'],
            dataset,
            device,
            dataloaders['dev'],
            eval_pipeline,
            print_result=False
        )
        current_metric = results.named_results["C-Index"]

        if current_metric > best_metric:
            best_metric = current_metric
            best_state = {key: model.state_dict() for key, model in models.items()}
            counter = 0  # Reset patience counter
        else:
            counter += 1

        print(
            f"Epoch {epoch}, C-Index: {current_metric:.4f}, Best: {best_metric:.4f}, Patience: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered. Restoring best model.")
            for key in models:
                models[key].load_state_dict(best_state[key])
            break

        torch.save(models['mirai'].state_dict(), f"mirai_chile/checkpoints/mirai_{dataset}_base_{epoch}.pt")
