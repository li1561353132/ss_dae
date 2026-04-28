import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 加载数据 =====================
df = pd.read_csv("../data/BSM1SUNNY.csv")
X = df.iloc[:, :35].values

# 标准化（必须）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled)


# ===================== 噪声函数 =====================
def add_gaussian_noise(x, noise_factor=0.25):  # 噪声提高！
    noise = torch.randn_like(x) * noise_factor
    return x + noise


# ===================== 强约束损失函数（三变量） =====================
def custom_denoise_loss(X1, X, X_noisy, lam=20.0):
    mse_recon = nn.functional.mse_loss(X1, X)
    mse_noisy = nn.functional.mse_loss(X_noisy, X)
    penalty = torch.clamp(mse_recon - mse_noisy, min=0.0)
    total_loss = mse_recon + lam * penalty
    return total_loss, mse_recon, mse_noisy


# ===================== 更强的自编码器（更深、更宽） =====================
class PowerfulDAE(nn.Module):
    def __init__(self, in_dim=35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, in_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ===================== 训练配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PowerfulDAE().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

EPOCHS = 300
NOISE_FACTOR = 0.25
LAMBDA = 20

X = X_tensor.to(device)

# ===================== 训练 =====================
print("开始训练……\n")
for epoch in range(EPOCHS):
    model.train()
    X_noisy = add_gaussian_noise(X, NOISE_FACTOR)
    X1 = model(X_noisy)
    loss, mse_r, mse_n = custom_denoise_loss(X1, X, X_noisy, LAMBDA)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch + 1:3d} | MSE_recon={mse_r:.6f} | MSE_noisy={mse_n:.6f} | 重构更好? {mse_r < mse_n}")

# ===================== 测试 & 可视化 =====================
model.eval()
with torch.no_grad():
    X_noisy = add_gaussian_noise(X, NOISE_FACTOR)
    X1 = model(X_noisy)

X_np = X.cpu().numpy()
Xn_np = X_noisy.cpu().numpy()
X1_np = X1.cpu().numpy()

# 计算平均MSE
mse_noisy = np.mean((X_np - Xn_np) ** 2)
mse_recon = np.mean((X_np - X1_np) ** 2)

print("\n" + "=" * 50)
print(f"平均 MSE(X,X_noisy) = {mse_noisy:.6f}")
print(f"平均 MSE(X,X1)      = {mse_recon:.6f}")
print(f"✅ 重构效果更好：{mse_recon < mse_noisy}")
print("=" * 50)

# 画图
feat_idx = 0
plt.figure(figsize=(14, 5))
plt.plot(X_np[:, feat_idx], label='原始 X', linewidth=1.5)
plt.plot(Xn_np[:, feat_idx], label='加噪 X_noisy', alpha=0.6)
plt.plot(X1_np[:, feat_idx], label='重构 X1', linewidth=2)
plt.title(f'去噪重构对比 | 特征：{df.columns[feat_idx]}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()