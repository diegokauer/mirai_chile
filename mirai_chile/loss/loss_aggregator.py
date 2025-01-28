import torch.nn as nn


class LossAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_functions = []

    def add_loss(self, loss_function):
        self.loss_functions.append(loss_function)

    def forward(self, logit, pmf, s, t, d):
        loss = [l(logit, pmf, s, t, d).item() for l in self.loss_functions]
        return loss
