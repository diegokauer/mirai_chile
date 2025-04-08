import argparse
import os
from socket import gethostname

import torch
import torch.distributed as dist

from mirai_chile.configs.mirai_base_config import TrainEnd2End
from mirai_chile.data.png_dataset import PNGRawDataset
from mirai_chile.loss.discriminator_loss import DiscriminationLoss
from mirai_chile.loss.mirai_loss import MiraiLoss
from mirai_chile.models.cumulative_probability_layer import CumulativeProbabilityLayer
from mirai_chile.models.discriminator import Discriminator
from mirai_chile.models.mirai_model import MiraiChile


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(args):
    epochs = args.epochs
    seed = args.seed
    dry_run = args.dry_run
    save_model = args.save_model
    save_each_epoch = args.save_each_epoch
    batch_size = args.batch_size

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    args = TrainEnd2End()
    head = CumulativeProbabilityLayer(612, args)
    model = MiraiChile(args=args, head=head)
    discriminator = Discriminator(5, 612)
    model.to_device(local_rank)
    discriminator.to_device(local_rank)

    models = {
        'mirai': model,
        'discriminator': discriminator
    }
    loss_functions = {
        'mirai': MiraiLoss(args=args).to_device(local_rank),
        'discriminator': DiscriminationLoss(args=args).to_device(local_rank),
    }

    print("Loading Datasets...")
    dataset = PNGRawDataset(
        png_table_path="./mirai_chile/data/dataset/png_dataset_example.csv",
        outcomes_table_path="./mirai_chile/data/dataset/outcomes.csv"
    )
    train_dataset = dataset.get_split("train")
    # dev_dataset = dataset.get_split("dev")
    # test_dataset = dataset.get_split("test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model")
    parser.add_argument('data_directory', type=str, help="path of the directory of files")
    parser.add_argument('--epochs', type=int, help="number of epochs", default=15)
    parser.add_argument('--seed', type=int, help="Random seed", default=1999)
    parser.add_argument('--batch_size', type=int, help="Batch size of dataloaders", default=64)
    parser.add_argument('--dry-run', type=bool, help="Dry run model", default=False)
    parser.add_argument('--save-model', type=bool, help="Save model", default=True)
    parser.add_argument('--save-each-epoch', type=bool, help="Save model on each epoch", default=True)
    # parser.add_argument()
    args = parser.parse_args()
    main(args)
