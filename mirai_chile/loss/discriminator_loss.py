import torch
import torch.nn.functional as F

from mirai_chile.configs.abstract_config import AbstractConfig
from mirai_chile.loss.abstract_loss import AbstractLoss


class DiscriminationLoss(AbstractLoss):
    def __init__(self, args=AbstractConfig()):
        super().__init__()
        self.args = args

    def forward(self, batch):
        device_logit, device = batch['device_logit'], batch['device'],
        adv_loss_per_sample = F.cross_entropy(device_logit, device, reduction='mean')
        adv_loss = torch.sum(adv_loss_per_sample) / device_logit.shape[0]
        gen_loss = -adv_loss
        return gen_loss, adv_loss
