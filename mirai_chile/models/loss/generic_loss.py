import torch
from torch import nn
import torch.nn.functional as F

from mirai_chile.configs.generic_config import GenericConfig


class GenericLoss(nn.Module):
    def __init__(self, args=GenericConfig()):
        super().__init__()
        self.args = args

    def forward(self, logit, pmf, s, t, d):
        pass

    def to_device(self, device):
        self.args.device = device
        self.to(device)
        return self