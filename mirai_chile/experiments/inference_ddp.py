import argparse
import os
from socket import gethostname

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mirai_chile.configs.abstract_config import AbstractConfig
from mirai_chile.configs.mirai_base_config import MiraiBaseConfigEval
from mirai_chile.data.generate_dataset import create_sampler, create_dataloader, PNGDataset
from mirai_chile.experiments.utils.csv import combine_csv
from mirai_chile.models.cumulative_probability_layer import CumulativeProbabilityLayer
from mirai_chile.models.mirai_model import MiraiChile


def infer(model, device, dataloader, rank):
    model.eval()

    logits_table = []
    transformer_table = []
    encoder_table = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            try:
                identifier = data["identifier"]
                data["images"].to(device)
                for key, val in data["batch"].items():
                    data["batch"][key] = val.to(device)

                logits, transformer_hidden, encoder_hidden = model(data["images"], data["batch"])

                for i, id in enumerate(identifier):
                    logits_table.append(
                        {"identifier": id, **{f"logit_{j}": logits[i, j].item() for j in range(logits.size(1))}})
                    transformer_table.append(
                        {"identifier": id,
                         **{f"hidden_{j}": transformer_hidden[i, j].item() for j in range(transformer_hidden.size(1))}}
                    )
                    encoder_table.append(
                        {"identifier": id,
                         **{f"encoder_{j}": encoder_hidden[i, j].item() for j in range(encoder_hidden.size(1))}}
                    )

                del data  # Free memory
                print(f"Inference completed on process")
            except Exception as e:
                print(data["identifier"], "failed:", e)

    pd.DataFrame(logits_table).to_csv(os.path.join(args.result_dir, f"logits_rank_{rank}.csv"), index=False)
    pd.DataFrame(transformer_table).to_csv(os.path.join(args.result_dir, f"transformer_hidden_rank_{rank}.csv"),
                                           index=False)
    pd.DataFrame(encoder_table).to_csv(os.path.join(args.result_dir, f"encoder_hidden_rank_{rank}.csv"), index=False)


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(args):
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

    dataset = PNGDataset(args.data_directory, AbstractConfig())
    sampler = create_sampler(dataset, AbstractConfig(), **sampler_kwargs)
    kwargs.update({"sampler": sampler})
    dataloader = create_dataloader(dataset, AbstractConfig(), **kwargs)

    model_args = MiraiBaseConfigEval()
    model_args.device = local_rank
    model = MiraiChile(model_args, CumulativeProbabilityLayer)
    model.to_device(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    infer(ddp_model, local_rank, dataloader, rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to infer data and save logits, and hidden vectors by examination")
    parser.add_argument('data_directory', type=str, help="path of the directory of files")
    parser.add_argument('--result_dir', type=str, help="Directory for the outputs", default=".")
    args = parser.parse_args()
    main(args)
    combine_csv(args.result_dir, "logits")
    combine_csv(args.result_dir, "transformer_hidden")
    combine_csv(args.result_dir, "encoder_hidden")
