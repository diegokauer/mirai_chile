import argparse
import os

import torch
import pandas as pd

from mirai_chile.models.mirai_model import MiraiChile
from mirai_chile.models.cumulative_probability_layer import Cumulative_Probability_Layer
from mirai_chile.configs.mirai_base_config import MiraiBaseConfigEval
from mirai_chile.configs.generic_config import GenericConfig
from mirai_chile.data.generate_dataset import create_dataloader

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_args = MiraiBaseConfigEval()
    model_args.device = device
    model = MiraiChile(model_args, Cumulative_Probability_Layer)
    model.eval()
    device = torch.device(device)
    model.to(device)

    inference_kwargs = {
        'batch_size': 1,
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    }

    dataloader = create_dataloader(args.data_directory, GenericConfig(), batch_size=1, **inference_kwargs)

    logits_table = []
    transformer_table = []
    encoder_table = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i == args.n_obs - 1:
                break

            identifier = data["identifier"]
            data["images"].to(device)
            for key, val in data["batch"].items():
                data["batch"][key] = val.to(device)

            logits, transformer_hidden, encoder_hidden = model(data["images"], data["batch"])

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

    pd.DataFrame(logits_table).to_csv(os.path.join(args.result_dir, f"logits_rank_test.csv"), index=False)
    pd.DataFrame(transformer_table).to_csv(os.path.join(args.result_dir, f"transformer_hidden_rank_test.csv"), index=False)
    pd.DataFrame(encoder_table).to_csv(os.path.join(args.result_dir, f"encoder_hidden_rank_test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to infer data and save logits, and hidden vectors by examination")
    parser.add_argument('data_directory', type=str, help="path of the directory of files")
    parser.add_argument('--result_dir', type=str, help="Directory for the outputs", default=".")
    parser.add_argument('--n_obs', type=int, help="Number of observations to be infered", default=10)
    args = parser.parse_args()
    main(args)