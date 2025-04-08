import torch.nn as nn

NUM_DEVICES = 8  # Number of manufacturers


class Discriminator(nn.Module):
    """
        Simple MLP discriminator
        Source: https://github.com/yala/Mirai/blob/master/onconet/models/discriminator.py
    """

    def __init__(self, num_logits=5, hidden_dim=612):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + num_logits, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, NUM_DEVICES)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.bn1(self.fc1(x)))
        hidden = self.relu(self.bn2(self.fc2(hidden)))
        z = self.fc3(hidden)
        return z

    def to_device(self, device):
        self.to(device)
