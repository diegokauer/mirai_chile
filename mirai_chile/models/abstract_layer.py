import torch.nn as nn

from mirai_chile.configs.abstract_config import AbstractConfig


class AbstractLayer(nn.Module):
    def __init__(self, args=AbstractConfig()):
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
