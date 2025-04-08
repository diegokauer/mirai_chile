import argparse
from collections import Counter

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from mirai_chile.configs.mirai_base_config import TrainEnd2End
from mirai_chile.data.png_dataset import PNGRawDataset
from mirai_chile.loss.discriminator_loss import DiscriminationLoss
from mirai_chile.loss.mirai_loss import MiraiLoss
from mirai_chile.model_evaluation.evaluation_functions.c_index import CIndex
from mirai_chile.model_evaluation.evaluation_functions.yearly_roc_auc import YearlyROCAUCFunction
from mirai_chile.model_evaluation.evaluation_functions.yearly_roc_auc_manufacturer import \
    YearlyROCAUCManufacturerFunction
from mirai_chile.model_evaluation.evaluation_pipeline import EvaluationPipeline
from mirai_chile.models.cumulative_probability_layer import CumulativeProbabilityLayer
from mirai_chile.models.discriminator import Discriminator
from mirai_chile.models.mirai_model import MiraiChile
from mirai_chile.train import train_model


def main(args):
    epochs = args.epochs
    seed = args.seed
    dry_run = args.dry_run
    save_model = args.save_model
    save_each_epoch = args.save_each_epoch
    batch_size = args.batch_size

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    args = TrainEnd2End()
    head = CumulativeProbabilityLayer(612, args)
    model = MiraiChile(args=args, head=head)
    discriminator = Discriminator(5, 612)
    model.to_device(device)
    discriminator.to_device(device)

    models = {
        'mirai': model,
        'discriminator': discriminator
    }
    loss_functions = {
        'mirai': MiraiLoss(args=args).to_device(device),
        'discriminator': DiscriminationLoss(args=args).to_device(device),
    }

    print("Loading Datasets...")
    dataset = PNGRawDataset(
        png_table_path="./mirai_chile/data/dataset/png_dataset_example.csv",
        outcomes_table_path="./mirai_chile/data/dataset/outcomes.csv"
    )
    train_dataset = dataset.get_split("train")
    dev_dataset = dataset.get_split("dev")
    test_dataset = dataset.get_split("test")

    del dataset

    labels = train_dataset.dataframe.cancer
    class_counts = Counter(labels)

    # Compute class weights (inverse frequency)
    num_samples = len(labels)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}

    # Convert class weights to sample weights
    sample_weights = np.array([class_weights[label] for label in labels])
    print("Class Weights:", sample_weights)

    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(weights=sample_weights_tensor,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_kwargs = {
        # 'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
        "batch_size": batch_size,
        # "shuffle": True
        "sampler": sampler
    }
    train_dataloader = DataLoader(train_dataset, **train_kwargs)

    test_kwargs = {
        # 'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
        "batch_size": batch_size,
        "shuffle": True
    }
    dev_dataloader = DataLoader(dev_dataset, **test_kwargs)
    test_dataloader = DataLoader(test_dataset, **test_kwargs)

    dataloaders = {
        'train': train_dataloader,
        'dev': dev_dataloader,
        'test': test_dataloader
    }

    optimizers = {
        'mirai': optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5),
        'discriminator': optim.Adam(discriminator.parameters())
    }

    eval_pipe = EvaluationPipeline()
    eval_pipe.add_metric(YearlyROCAUCFunction(), {"max_followup": 5})
    eval_pipe.add_metric(YearlyROCAUCManufacturerFunction(), {"max_followup": 5})
    eval_pipe.add_metric(CIndex(), {"censoring_dist": dataset.censoring_dist, "max_followup": 5})

    print("Beginning training...")
    train_model(
        models=models,
        loss_functions=loss_functions,
        dataset="transformer_hidden",
        device=device,
        dataloaders=dataloaders,
        optimizers=optimizers,
        epochs=epochs,
        eval_pipeline=eval_pipe,
        dry_run=dry_run
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model")
    # parser.add_argument('data_directory', type=str, help="path of the directory of files")
    parser.add_argument('--epochs', type=int, help="number of epochs", default=15)
    parser.add_argument('--seed', type=int, help="Random seed", default=1999)
    parser.add_argument('--batch_size', type=int, help="Batch size of dataloaders", default=64)
    parser.add_argument('--dry-run', type=bool, help="Dry run model", default=False)
    parser.add_argument('--save-model', type=bool, help="Save model", default=True)
    parser.add_argument('--save-each-epoch', type=bool, help="Save model on each epoch", default=True)
    # parser.add_argument()
    args = parser.parse_args()
    main(args)
