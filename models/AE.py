import torch
import torch.nn as nn

class AE_OnlyReg(nn.Module):
    def __init__(self, input_dim=35, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.regressor(z)
        return y_pred