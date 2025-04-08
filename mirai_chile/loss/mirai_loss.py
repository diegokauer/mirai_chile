import torch
from torch import nn

from mirai_chile.loss.abstract_loss import AbstractLoss


class MiraiLoss(AbstractLoss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss = nn.functional.binary_cross_entropy_with_logits

    def forward(self, batch):
        """
        Vectorized forward for the MiraiLoss function.
        :param logits: Tensor of shape (B, TIME_MAX), logits of the MTLR classifier.
        :param t: Tensor of shape (B,), time until observed.
        :param d: Tensor of shape (B,), binary 1 if event is observed, 0 if censored.
        :return: Scalar loss value for the batch.
        """
        logit, pmf, time_to_event, cancer = batch['logit'], batch['pmf'], batch['time_to_event'], batch['cancer']

        device = logit.device
        batch_size = logit.size(0)

        # Create masks based on `t` and `d`
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # Shape (B, 1)
        time_indices = torch.arange(self.args.max_followup, device=device).unsqueeze(0)  # Shape (1, T)

        # Create the `y` mask for event occurrence
        y = (time_indices >= time_to_event.unsqueeze(1)).float()  # Shape (B, T), fills with 1 after time `t`

        # Create the `y_mask` for censoring
        y_mask = torch.ones_like(y)  # Start with all ones
        y_mask[cancer == 0] = (time_indices < time_to_event.unsqueeze(1))[
            cancer == 0].float()  # Zero out after time `t` for censored

        # Compute binary cross-entropy loss with mask
        return self.loss(logit, y.float(), weight=y_mask.float(), reduction='sum') / torch.sum(y_mask.float())
