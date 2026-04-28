import torch
import torch.nn as nn

class BP_Net(nn.Module):
    def __init__(self, input_dim=35):
        super().__init__()
        self.bp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.bp(x)
