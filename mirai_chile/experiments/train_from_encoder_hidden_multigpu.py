import argparse
import os
from socket import gethostname

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from mirai_chile.configs.train_config import TrainEncoderHiddenConfig
from mirai_chile.data.encoder_hidden import EncoderHiddenDataset
from mirai_chile.loss.pmf_loss import PMFLoss
from mirai_chile.model_evaluation.evaluation_functions.yearly_roc_auc import YearlyROCAUCFunction
from mirai_chile.model_evaluation.evaluation_functions.yearly_roc_auc_manufacturer import \
    YearlyROCAUCManufacturerFunction
from mirai_chile.model_evaluation.evaluation_pipeline import EvaluationPipeline
from mirai_chile.models.mirai_model import MiraiChile
from mirai_chile.models.pmf_layer import PMFLayer
from mirai_chile.predict import predict_probas
from mirai_chile.test import test_model
from mirai_chile.train import train_model


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(args):
    epochs = args.epochs
    seed = args.seed
    dry_run = args.dry_run
    save_model = args.save_model
    save_each_epoch = args.save_each_epoch

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    kwargs = {
        'batch_size': 1,
        'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
        'pin_memory': True,
    }
    sampler_kwargs = {
        "num_replicas": world_size,
        "rank": rank,
    }

    print("Loading Datasets...")
    dataset = EncoderHiddenDataset(
        encoder_hidden_table_path="mirai_chile/data/dataset/combined_encoder_hidden.csv",
        outcomes_table_path="mirai_chile/data/dataset/outcomes.csv"
    )
    train_dataset = dataset.get_split("train")
    dev_dataset = dataset.get_split("dev")
    test_dataset = dataset.get_split("test")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_kwargs = {
        'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
        "batch_size": 32,
        "pin_memory": True,
        "sampler": train_sampler
    }
    train_dataloader = DataLoader(train_dataset, **train_kwargs)

    test_kwargs = {
        'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
        "batch_size": 32,
        "shuffle": True,
        "pin_memory": True
    }
    dev_dataloader = DataLoader(dev_dataset, **test_kwargs)
    test_dataloader = DataLoader(test_dataset, **test_kwargs)

    args = TrainEncoderHiddenConfig()
    loss_function = PMFLoss(args)
    head = PMFLayer(612, args)
    model = MiraiChile(args=args, loss_function=loss_function, head=head)
    model.to_device(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-5)

    eval_pipe = EvaluationPipeline()
    eval_pipe.add_metric(YearlyROCAUCFunction(), {"max_followup": 5})
    eval_pipe.add_metric(YearlyROCAUCManufacturerFunction(), {"max_followup": 5})

    print("Beginning training...")
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_model(ddp_model, "encoder_hidden", local_rank, train_dataloader, optimizer, epoch, dry_run)
        test_model(ddp_model, "encoder_hidden", local_rank, DataLoader(dataset, **test_kwargs), eval_pipe, dry_run)
        if save_each_epoch and save_model:
            torch.save(model.state_dict(), f"mirai_chile/checkpoints/mirai_encoder_pmf_epoch_{epoch}_{rank}_mp.pt")

    if save_model:
        torch.save(model.state_dict(), f"mirai_chile/checkpoints/mirai_encoder_pmf_final_{rank}_mp.pt")

    print("Predicting future cancer probabilities...")
    prob_df = predict_probas(model, "encoder_hidden", local_rank, DataLoader(dataset, **test_kwargs), dry_run=dry_run)
    eval_pipe.flush()
    eval_pipe.eval_dataset(prob_df)
    print(eval_pipe)

    dist.destroy_process_group()


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
