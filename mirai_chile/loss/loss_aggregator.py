import torch.nn as nn


# TODO: Add a lambda for each loss term and return the summation

class LossAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_functions = []

    def add_loss(self, loss_function):
        self.loss_functions.append(loss_function)

    def forward(self, batch):
        loss = [l(batch).item() for l in self.loss_functions]
        return loss
