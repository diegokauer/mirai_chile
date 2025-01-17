import torch
import torch.nn as nn


class PMFLayer(nn.Module):
    def __init__(self, num_features, args):
        super(PMFLayer, self).__init__()
        max_followup = args.max_followup
        self.args = args
        self.pmf = nn.Linear(num_features, max_followup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hidden = self.pmf(x)
        pmf = self.relu(hidden)
        return pmf