import torch
from torch import nn
import torch.nn.functional as F

from mirai_chile.models.loss.generic_loss import GenericLoss


class PMFLoss(GenericLoss):
    def __init__(self, args=None):
        super().__init__()
        if not (args is None):
            self.args = args
        # Difference matrix for Mirai Backbone
        dif_matrix = torch.zeros((self.args.max_followup, self.args.max_followup), dtype=torch.float32)
        dif_matrix[0, 0] = 1
        for i in range(1, self.args.max_followup):
            dif_matrix[i, i - 1] = -1
            dif_matrix[i, i] = 1
        self.register_buffer('dif_matrix', dif_matrix)

        # Lower triangular matrix without diagonal
        l_t_mat = torch.tril(torch.ones([self.args.max_followup, self.args.max_followup], dtype=torch.float32),
                             diagonal=-1)
        self.register_buffer('lower_triangular_matrix_no_diag', l_t_mat)

        # Ones vector for cumulative probability calculation
        self.register_buffer('ones', torch.ones(self.args.max_followup, dtype=torch.float32))

    def forward(self, logit, pmf, s, t, d):
        """
        logit: Tensor of shape (B, T)
        t: Indices for the time step, shape (B,)
        d: Indicators for event occurrence, shape (B,)
        """
        device = self.args.device

        # Ensure t is used correctly
        batch_indices = torch.arange(logit.size(0), device=logit.device)
        pmf_t = pmf[batch_indices, t]
        s_t = s[batch_indices, t]

        # Numerical stability in log computation
        loss = d * torch.log(torch.clamp(pmf_t, min=1e-9)) + (1 - d) * torch.log(torch.clamp(s_t, min=1e-9))
        return -torch.mean(loss)  # Negative log likelihood