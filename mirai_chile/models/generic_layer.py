import torch
import torch.nn as nn
import torch.nn.functional as F

from mirai_chile.configs.generic_config import GenericConfig


class GenericLayer(nn.Module):
    def __init__(self, args=GenericConfig()):
        super().__init__()
        self.args = args

    def forward(self, x):
        pass

    def logit_to_cancer_prob(self, logit):
        pass

    def to_device(self, device):
        self.args.device = device
        self.to(device)
        return self