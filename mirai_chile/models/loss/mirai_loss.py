import torch
from mirai_chile.models.loss.abstract_loss import AbstractLoss
from torch import nn


class MiraiLoss(AbstractLoss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss = nn.functional.binary_cross_entropy_with_logits

    def forward(self, logits, pmf, s, t, d):
        """
        Vectorized forward for the MiraiLoss function.
        :param logits: Tensor of shape (B, TIME_MAX), logits of the MTLR classifier.
        :param t: Tensor of shape (B,), time until observed.
        :param d: Tensor of shape (B,), binary 1 if event is observed, 0 if censored.
        :return: Scalar loss value for the batch.
        """
        device = logits.device
        batch_size = logits.size(0)

        # Create masks based on `t` and `d`
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # Shape (B, 1)
        time_indices = torch.arange(self.args.max_followup, device=device).unsqueeze(0)  # Shape (1, T)

        # Create the `y` mask for event occurrence
        y = (time_indices >= t.unsqueeze(1)).float()  # Shape (B, T), fills with 1 after time `t`

        # Create the `y_mask` for censoring
        y_mask = torch.ones_like(y)  # Start with all ones
        y_mask[d == 0] = (time_indices < t.unsqueeze(1))[d == 0].float()  # Zero out after time `t` for censored

        # Compute binary cross-entropy loss with mask
        return self.loss(logits, y.float(), weight=y_mask.float(), size_average=False) / torch.sum(y_mask.float())
