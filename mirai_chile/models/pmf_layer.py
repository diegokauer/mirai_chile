import torch
import torch.nn as nn
import torch.nn.functional as F

from mirai_chile.models.abstract_layer import AbstractLayer


class PMFLayer(AbstractLayer):
    def __init__(self, num_features, args=None):
        super().__init__()
        if not (args is None):
            self.args = args
        self.pmf = nn.Linear(num_features, self.args.max_followup)
        self.dropout = nn.Dropout(self.args.dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        pmf = self.pmf(x)
        # pmf = self.dropout(hidden)
        return pmf

    def logit_to_cancer_prob(self, logit):
        """
        Converts logits to cancer survival probabilities.

        :param logits: A tensor of shape (B, T) where B is the batch size and T is the number of time intervals.
        :return: A tensor representing the survival probabilities.
        """
        device = self.args.device
        B, N = logit.size()

        # Append a zero column to logits
        logit = torch.cat((logit, torch.zeros(B, 1, device=device)), dim=1)
        N += 1

        # Create lower triangular matrix without diagonal
        l_t_mat = torch.tril(torch.ones(N, N, device=device), diagonal=-1)

        # Compute PMF (Probability Mass Function)
        pmf = F.softmax(logit, dim=1)  # Exclude the last column

        # Compute survival probabilities
        s = torch.matmul(pmf, l_t_mat)

        return pmf[:, :-1], s[:, :-1]
