import torch
import torch.nn as nn

class DAE_SoftSensor(nn.Module):
    def __init__(self, input_dim=35, latent_dim=8):
        super(DAE_SoftSensor, self).__init__()

        # 编码器：降维+去噪+特征提取
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)  # 隐层特征Z
        )

        # 解码器：重构去噪后的输入
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        # 回归头：软测量预测BODe
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    # 加噪声函数（DAE核心）
    def add_noise(self, x, noise_factor=0.01):
        noise = torch.randn_like(x) * noise_factor
        x_noisy = x + noise
        return x_noisy

    def forward(self, x):
        # 编码
        z = self.encoder(x)
        # 解码重构
        x_recon = self.decoder(z)
        # 回归预测
        y_pred = self.regressor(z)
        return x_recon, y_pred

