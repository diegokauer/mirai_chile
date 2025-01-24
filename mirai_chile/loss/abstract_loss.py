from torch import nn

from mirai_chile.configs.abstract_config import AbstractConfig


class AbstractLoss(nn.Module):
    def __init__(self, args=AbstractConfig()):
        super().__init__()
        self.args = args

    def forward(self, logit, pmf, s, t, d):
        pass

    def to_device(self, device):
        self.args.device = device
        self.to(device)
        return self
