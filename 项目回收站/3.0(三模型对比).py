import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ===================== 1. 设备配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===================== 2. 数据加载与预处理 =====================
df = pd.read_csv("../data/BSM1SUNNY.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

total_len = len(X_scaled)
size_train = int(0.7 * total_len)
size_val = int(0.1 * total_len)
size_test = total_len - size_train - size_val

X_train = X_scaled[:size_train]
y_train = y_scaled[:size_train]
X_val = X_scaled[size_train : size_train+size_val]
y_val = y_scaled[size_train : size_train+size_val]
X_test = X_scaled[size_train+size_val : ]
y_test = y_scaled[size_train+size_val : ]

batch_size = 32
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===================== 模型1：DAE去噪自编码器 =====================
class DAE_SoftSensor(nn.Module):
    def __init__(self, input_dim=35, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )

    def add_noise(self, x, noise_factor=0.01):
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        y_pred = self.regressor(z)
        return x_recon, y_pred

# ===================== 模型2：纯回归AE（无解码器） =====================
class AE_OnlyReg(nn.Module):
    def __init__(self, input_dim=35, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.regressor(z)
        return y_pred

# ===================== 模型3：BP神经网络 =====================
class BP_Net(nn.Module):
    def __init__(self, input_dim=35):
        super().__init__()
        self.bp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.bp(x)

# ===================== 训练函数 =====================
def train_model(model, model_type, epochs=200, lr=0.0005):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    print(f"\n===== 训练 {model_type} 模型 =====")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if model_type == "DAE":
                x_noisy = model.add_noise(batch_x)
                x_recon, y_pred = model(x_noisy)
                loss_pred = criterion(y_pred, batch_y)
                loss_recon = criterion(x_recon, batch_x)
                loss = 1 * loss_pred + 0.1 * loss_recon
            else:
                y_pred = model(batch_x)
                loss = criterion(y_pred, batch_y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                if model_type == "DAE":
                    _, y_pred = model(batch_x)
                else:
                    y_pred = model(batch_x)
                val_loss += criterion(y_pred, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | 训练损失: {train_loss:.5f} | 验证损失: {val_loss:.5f}")

    return train_losses, val_losses

# ===================== 测试函数（支持加噪） =====================
def test_with_noise(model, model_type, noise_factor=0.0):
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            if noise_factor > 0:
                batch_x = batch_x + torch.randn_like(batch_x) * noise_factor

            if model_type == "DAE":
                _, y_pred = model(batch_x)
            else:
                y_pred = model(batch_x)
            y_pred_list.append(y_pred.cpu().numpy())
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = scaler_y.inverse_transform(y_test)
    y_pred = scaler_y.inverse_transform(y_pred)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return y_true, y_pred, r2, rmse, mae

# ===================== 初始化并训练三个模型 =====================
dae_model = DAE_SoftSensor(input_dim=35, latent_dim=8).to(device)
ae_reg_model = AE_OnlyReg(input_dim=35, latent_dim=8).to(device)
bp_model = BP_Net(input_dim=35).to(device)
epochs = 150
train_model(dae_model, "DAE", epochs=epochs)
train_model(ae_reg_model, "AE_Reg", epochs=epochs)
train_model(bp_model, "BP", epochs=epochs)

# ===================== 噪声等级设置 =====================
noise_levels = {
    "无噪声": 0.0,
    "低噪声": 0.05,
    "高噪声": 0.15
}

# 存储所有结果
results = {}
y_true_global = None

# ===================== 批量测试 =====================
for name, noise in noise_levels.items():
    print(f"\n{'='*60}")
    print(f"               {name} 测试 (noise={noise})")
    print(f"{'='*60}")

    yt, dae_p, dae_r2, dae_rmse, dae_mae = test_with_noise(dae_model, "DAE", noise)
    _, ae_p, ae_r2, ae_rmse, ae_mae = test_with_noise(ae_reg_model, "AE_Reg", noise)
    _, bp_p, bp_r2, bp_rmse, bp_mae = test_with_noise(bp_model, "BP", noise)

    if y_true_global is None:
        y_true_global = yt

    results[name] = {
        "dae_pred": dae_p, "dae_r2": dae_r2, "dae_rmse": dae_rmse,
        "ae_pred": ae_p, "ae_r2": ae_r2, "ae_rmse": ae_rmse,
        "bp_pred": bp_p, "bp_r2": bp_r2, "bp_rmse": bp_rmse,
    }

    print(f"DAE\t R²={dae_r2:.4f}\t RMSE={dae_rmse:.4f}")
    print(f"纯回归AE\t R²={ae_r2:.4f}\t RMSE={ae_rmse:.4f}")
    print(f"BP\t R²={bp_r2:.4f}\t RMSE={bp_rmse:.4f}")

# ===================== 精度下降对比 =====================
print(f"\n{'='*70}")
print(f"                        抗干扰能力对比（R²下降）")
print(f"{'='*70}")

r2_dae_no_noise = results["无噪声"]["dae_r2"]
r2_ae_no_noise = results["无噪声"]["ae_r2"]
r2_bp_no_noise = results["无噪声"]["bp_r2"]

r2_dae_low = results["低噪声"]["dae_r2"]
r2_ae_low = results["低噪声"]["ae_r2"]
r2_bp_low = results["低噪声"]["bp_r2"]

r2_dae_high = results["高噪声"]["dae_r2"]
r2_ae_high = results["高噪声"]["ae_r2"]
r2_bp_high = results["高噪声"]["bp_r2"]

print(f"无噪声基准：DAE={r2_dae_no_noise:.4f} | AE={r2_ae_no_noise:.4f} | BP={r2_bp_no_noise:.4f}")
print(f"低噪声下降：DAE={r2_dae_no_noise - r2_dae_low:.4f} | AE={r2_ae_no_noise - r2_ae_low:.4f} | BP={r2_bp_no_noise - r2_bp_low:.4f}")
print(f"高噪声下降：DAE={r2_dae_no_noise - r2_dae_high:.4f} | AE={r2_ae_no_noise - r2_ae_high:.4f} | BP={r2_bp_no_noise - r2_bp_high:.4f}")

# ===================== 绘图：3张对比图 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1：无噪声
plt.figure(figsize=(11, 6))
plt.plot(y_true_global, 'k-', label='真实值', lw=2.5)
plt.plot(results["无噪声"]["dae_pred"], 'r-', label=f'DAE  R²={results["无噪声"]["dae_r2"]:.4f}', lw=1.8)
plt.plot(results["无噪声"]["ae_pred"], 'g-', label=f'纯回归AE  R²={results["无噪声"]["ae_r2"]:.4f}', lw=1.8)
plt.plot(results["无噪声"]["bp_pred"], 'b-', label=f'BP  R²={results["无噪声"]["bp_r2"]:.4f}', lw=1.8)
plt.title("无噪声环境下三模型预测对比", fontsize=14)
plt.xlabel("时间步")
plt.ylabel("BODe浓度")
plt.legend()
plt.grid(True)

# 图2：低噪声
plt.figure(figsize=(11, 6))
plt.plot(y_true_global, 'k-', label='真实值', lw=2.5)
plt.plot(results["低噪声"]["dae_pred"], 'r-', label=f'DAE  R²={results["低噪声"]["dae_r2"]:.4f}', lw=1.8)
plt.plot(results["低噪声"]["ae_pred"], 'g-', label=f'纯回归AE  R²={results["低噪声"]["ae_r2"]:.4f}', lw=1.8)
plt.plot(results["低噪声"]["bp_pred"], 'b-', label=f'BP  R²={results["低噪声"]["bp_r2"]:.4f}', lw=1.8)
plt.title("低噪声环境下三模型预测对比", fontsize=14)
plt.xlabel("时间步")
plt.ylabel("BODe浓度")
plt.legend()
plt.grid(True)

# 图3：高噪声
plt.figure(figsize=(11, 6))
plt.plot(y_true_global, 'k-', label='真实值', lw=2.5)
plt.plot(results["高噪声"]["dae_pred"], 'r-', label=f'DAE  R²={results["高噪声"]["dae_r2"]:.4f}', lw=1.8)
plt.plot(results["高噪声"]["ae_pred"], 'g-', label=f'纯回归AE  R²={results["高噪声"]["ae_r2"]:.4f}', lw=1.8)
plt.plot(results["高噪声"]["bp_pred"], 'b-', label=f'BP  R²={results["高噪声"]["bp_r2"]:.4f}', lw=1.8)
plt.title("高噪声环境下三模型预测对比", fontsize=14)
plt.xlabel("时间步")
plt.ylabel("BODe浓度")
plt.legend()
plt.grid(True)

plt.show()