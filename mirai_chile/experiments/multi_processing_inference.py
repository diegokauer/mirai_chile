import argparse
import os

import torch
import torch.multiprocessing as mp
import pandas as pd

from mirai_chile.models.mirai_model import MiraiChile
from mirai_chile.models.cumulative_probability_layer import Cumulative_Probability_Layer
from mirai_chile.configs.mirai_base_config import MiraiBaseConfigEval
from mirai_chile.configs.generic_config import GenericConfig
from mirai_chile.data.generate_dataset import create_dataloader


def infer(rank, queue, result_dir):
    result_dir = os.path.expanduser(result_dir)
    device = torch.device(f"cuda:{rank}")

    args = MiraiBaseConfigEval()
    args.device = device
    (f"Inference with {device}")
    model = MiraiChile(args, Cumulative_Probability_Layer)
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    logits_table = []
    transformer_table = []
    encoder_table = []

    while True:
        data = queue.get()
        if data is None:  # check for sentinel value
            break

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
        print(f"Inference completed on process {rank}")

    pd.DataFrame(logits_table).to_csv(os.path.join(result_dir, f"logits_rank_{rank}.csv"), index=False)
    pd.DataFrame(transformer_table).to_csv(os.path.join(result_dir, f"transformer_hidden_rank_{rank}.csv"), index=False)
    pd.DataFrame(encoder_table).to_csv(os.path.join(result_dir, f"encoder_hidden_rank_{rank}.csv"), index=False)


def main(args):
    queue = mp.Queue()
    processes = []
    dataloader = create_dataloader(args.data_directory, GenericConfig(), batch_size=16)
    for rank in range(args.num_processes):
        p = mp.Process(target=infer, args=(rank, queue, args.result_dir))
        p.start()
        processes.append(p)
    for data in dataloader:
        queue.put(data)
    for _ in range(args.num_processes):
        queue.put(None)  # sentinel value to signal subprocesses to exit
    for p in processes:
        p.join()  # wait for all subprocesses to finish

    for table_name in ["logits", "transformer_hidden", "encoder_hidden"]:
        merged_file = os.path.join(args.result_dir, f"{table_name}_complete.csv")
        all_results = pd.concat(
            [pd.read_csv(os.path.join(args.result_dir, f"{table_name}_rank_{rank}.csv")) for rank in
             range(args.num_processes)],
            ignore_index=True,
        )
        all_results.to_csv(merged_file, index=False)
        print(f"Saved complete {table_name} table to {merged_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to infer data and save logits, and hidden vectors by examination")
    parser.add_argument('data_directory', type=str, help="path of the directory of files")
    parser.add_argument('num_processes', type=int, help="Number of GPUs used to process the files")
    parser.add_argument('result_dir', type=str, help="Directory for the outputs")
    args = parser.parse_args()
    main(args)
