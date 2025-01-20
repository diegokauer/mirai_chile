import argparse

import torch
import torch.optim as optim
from mirai_chile.configs.train_config import TrainEncoderHiddenConfig
from mirai_chile.data.encoder_hidden import EncoderHiddenDataset
from mirai_chile.model_evaluation.evaluation_functions.yearly_roc_auc import YearlyROCAUCFunction
from mirai_chile.model_evaluation.evaluation_pipeline import EvaluationPipeline
from mirai_chile.models.loss.pmf_loss import PMFLoss
from mirai_chile.models.mirai_model import MiraiChile
from mirai_chile.models.pmf_layer import PMFLayer
from mirai_chile.predict import predict_probas
from mirai_chile.test import test_model
from mirai_chile.train import train_model
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader


def main(args):
    epochs = args.epochs
    seed = args.seed
    dry_run = args.dry_run
    save_model = args.save_model
    save_each_epoch = args.save_each_epoch

    torch.manual_seed(seed)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    args = TrainEncoderHiddenConfig()
    loss_function = PMFLoss(args)
    head = PMFLayer(612, args)
    model = MiraiChile(args=args, loss_function=loss_function, head=head)
    model.to_device(device)

    print("Loading Datasets...")
    dataset = EncoderHiddenDataset(
        encoder_hidden_table_path="mirai_chile/data/dataset/combined_encoder_hidden.csv",
        outcomes_table_path="mirai_chile/data/dataset/outcomes.csv"
    )
    train_dataset = dataset.get_split("train")
    dev_dataset = dataset.get_split("dev")
    test_dataset = dataset.get_split("test")

    del dataset

    train_kwargs = {
        # "num_workers": 1,
        "batch_size": 32,
        "shuffle": True
    }
    train_dataloader = DataLoader(train_dataset, **train_kwargs)

    test_kwargs = {
        # "num_workers": 20,
        "batch_size": 32,
        "shuffle": True
    }
    dev_dataloader = DataLoader(dev_dataset, **test_kwargs)
    test_dataloader = DataLoader(test_dataset, **test_kwargs)

    optimizer = optim.Adam(model.parameters())
    scheduler = ExponentialLR(optimizer, 0.95)

    eval_pipe = EvaluationPipeline()
    eval_pipe.add_metric(YearlyROCAUCFunction(), {"max_followup": 5})

    print("Beginning training...")
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_model(model, "encoder_hidden", device, train_dataloader, optimizer, epoch, dry_run)
        test_model(model, "encoder_hidden", device, dev_dataloader, eval_pipe, dry_run)
        if save_each_epoch and save_model:
            torch.save(model.state_dict(), f"mirai_chile/checkpoints/mirai_logit_pmf_epoch_{epoch}.pt")
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), f"mirai_chile/checkpoints/mirai_logit_pmf_final.pt")

    print("Predicting future cancer probabilities...")
    predict_probas(model, "encoder_hidden", device, test_dataloader, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model")
    parser.add_argument('--epochs', type=int, help="number of epochs", default=10)
    parser.add_argument('--seed', type=int, help="Random seed", default=1999)
    parser.add_argument('--batch_size', type=int, help="Batch size of dataloaders", default=32)
    parser.add_argument('--dry-run', type=bool, help="Dry run model", default=False)
    parser.add_argument('--save-model', type=bool, help="Save model", default=True)
    parser.add_argument('--save-each-epoch', type=bool, help="Save model on each epoch", default=True)
    # parser.add_argument()
    args = parser.parse_args()
    main(args)
