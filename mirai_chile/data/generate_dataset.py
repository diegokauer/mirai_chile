import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from mirai_chile.data.pre_processing import pre_process_images
from mirai_chile.configs.generic_config import GenericConfig


# Assuming generate_data_dataframe is already defined
# Add this custom dataset class
class PNGDataset(Dataset):
    def __init__(self, directory, args):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the file paths.
            transform (callable, optional): Optional transform to apply on an image.
        """
        self.dataframe = self.generate_data_dataframe(directory)
        self.args = args

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load the images
        images = [row[col] for col in ["path1", "path2", "path3", "path4"]]

        try:
            images, batch = pre_process_images(images, self.args)
        except Exception as e:
            print(f"Skipping index {idx} due to processing error: {e}")
            raise RuntimeError(f"Index {idx} failed")  # PyTorch will skip this sample automatically.

        # Example: Return the images as a dictionary
        return {"images": images, "batch": batch, "identifier": row["index"]}

    def save_as_csv(self, csv_path):
        # Save to a CSV or print
        self.dataframe.to_csv(csv_path, index=False)

    @staticmethod
    def generate_data_dataframe(directory, save_as_csv=False, csv_path=None):
        directory = os.path.expanduser(directory)
        files = sorted([f for f in os.listdir(directory) if f.endswith(".png")])

        file_dict = {}
        for file in files:
            base_id = "_".join(file.split("_")[:2])  # Extract identifier
            if base_id not in file_dict:
                file_dict[base_id] = {"CC_L": None, "CC_R": None, "MLO_L": None, "MLO_R": None}

            if "CC_L" in file:
                file_dict[base_id]["CC_L"] = os.path.join(directory, file)
            elif "CC_R" in file:
                file_dict[base_id]["CC_R"] = os.path.join(directory, file)
            elif "MLO_L" in file:
                file_dict[base_id]["MLO_L"] = os.path.join(directory, file)
            elif "MLO_R" in file:
                file_dict[base_id]["MLO_R"] = os.path.join(directory, file)

        df = pd.DataFrame.from_dict(file_dict, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={"CC_L": "path1", "CC_R": "path2", "MLO_L": "path3", "MLO_R": "path4"}, inplace=True)

        return df.dropna().reset_index(drop=True)


# Function to create DataLoader
def create_dataloader(dataset, args=GenericConfig(), **kwargs):
    """
    Args:
        directory (str): Directory containing the .png files.
        transform (callable, optional): Optional transform to apply on images.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        save_as_csv (bool): Whether to save the DataFrame as a CSV file.
        csv_path (str): Path to save the CSV file if save_as_csv is True.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    # Generate the DataFrame

    # Create the dataset
    # dataset = PNGDataset(directory, args)

    # Create the DataLoader
    dataloader = DataLoader(dataset, **kwargs)

    return dataloader


def create_sampler(dataset, args=GenericConfig(), **kwargs):
    """
    Args:
        directory (str): Directory containing the .png files.
        transform (callable, optional): Optional transform to apply on images.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        save_as_csv (bool): Whether to save the DataFrame as a CSV file.
        csv_path (str): Path to save the CSV file if save_as_csv is True.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    # Generate the DataFrame

    # Create the dataset
    # dataset = PNGDataset(directory, args)

    # Create the DataLoader
    dds = DistributedSampler(dataset, **kwargs)

    return dds
