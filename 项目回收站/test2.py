import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ===================== 1. 设备配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===================== 2. 数据加载与预处理 =====================
# 加载数据
df = pd.read_csv("../data/BSM1SUNNY.csv")   # df.shape:(1343, 36)  <class 'pandas.core.frame.DataFrame'>

# 划分输入X(1-35列) 和 输出y(第36列 BODe)
X = df.iloc[:, :-1].values  # 前35列
y = df.iloc[:, -1].values  # 最后1列
y = y.reshape(-1, 1)  # 转为列向量
# print(X.shape, y.shape)                   # X:(1344, 35) y:(1344, 1)  <class 'numpy.ndarray'>

# 标准化     X,y -->  X_scaled,y_scaled
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
# print(X_scaled.shape,y_scaled.shape)      # X_scaled:(1344, 35) y_scaled:(1344, 1)  <class 'numpy.ndarray'>
# print(X),print(),print(),print(X_scaled),print(),print(),print(y),print(),print(),print(y_scaled)

# 时序数据划分（不打乱）   X_scaled,y_scaled -->  X_train, y_train 、 X_val, y_val 、 X_test, y_test
size_train = int(0.7 * len(X_scaled));    size_val = int(0.1 * len(X_scaled));    size_test = len(X_scaled) - size_train - size_val
X_train, y_train = X_scaled[:size_train], y_scaled[:size_train]
X_val, y_val = X_scaled[size_train:size_train + size_val], y_scaled[size_train:size_train + size_val]
X_test, y_test = X_scaled[-size_test:], y_scaled[-size_test:]
# <class 'numpy.ndarray'>  训练集: (940, 35),(940, 1)  验证集: (134, 35),(134, 1)  测试集: (270, 35),(270, 1)
# print(f"训练集: {X_train.shape},{y_train.shape};验证集: {X_val.shape},{y_val.shape};测试集: {X_test.shape},{y_test.shape}")

# 转换为Tensor     X_train, y_train --train_dataset-->  rain_loader ; X_val, y_val --val_dataset-->  val_loader ; X_test, y_test --test_dataset-->  test_loader
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))      # <class 'torch.utils.data.dataset.TensorDataset'>
# print(f"TensorDataset数据:{len(train_dataset), len(val_dataset), len(test_dataset)}")           # TensorDataset数据:(940, 134, 270)
# print(train_dataset),print(train_dataset[0])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)                    # <class 'torch.utils.data.dataloader.DataLoader'>
# print(f"DataLoader数据:{len(train_loader), len(val_loader), len(test_loader)}")                   # DataLoader数据:(30, 5, 9)
# print(train_loader)

# ===================== 3. 去噪自编码器(DAE)模型定义 =====================
class DAE_SoftSensor(nn.Module):
    def __init__(self, input_dim=35, latent_dim=16):
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
    def add_noise(self, x, noise_factor=0.05):
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



# ===================== 4. 模型初始化 ===================== 【保持不变】
model = DAE_SoftSensor(input_dim=35, latent_dim=16).to(device)
# print("模型子模块:"),print(model)
# print("encoder模块:"model._modules['encoder'])
# print("模型参数：")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

criterion_recon = nn.MSELoss()  # 重构损失
criterion_pred = nn.MSELoss()  # 预测损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 150
alpha = 1  # 预测损失权重，重构损失权重=0.2


# ===================== 5. 训练模型 ===================== 【修改为两个独立训练函数】
# ===================== 训练函数1：双加权损失训练 =====================
def train_dual_loss():
    print("\n========== 开始训练：双加权损失（重构+预测） ==========")
    model_dual = DAE_SoftSensor(input_dim=35, latent_dim=16).to(device)
    optimizer_dual = optim.Adam(model_dual.parameters(), lr=0.001)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model_dual.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_noisy = model_dual.add_noise(batch_x)
            x_recon, y_pred = model_dual(batch_x_noisy)

            loss_recon = criterion_recon(x_recon, batch_x)
            loss_pred = criterion_pred(y_pred, batch_y)
            loss = alpha * loss_pred + (1 - alpha) * loss_recon

            optimizer_dual.zero_grad()
            loss.backward()
            optimizer_dual.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model_dual.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                x_recon, y_pred = model_dual(batch_x)
                loss_recon = criterion_recon(x_recon, batch_x)
                loss_pred = criterion_pred(y_pred, batch_y)
                loss = alpha * loss_pred + (1 - alpha) * loss_recon
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[双损失] Epoch {epoch + 1}/{epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

    # 保存模型
    torch.save(model_dual.state_dict(), "dae_dual_loss_model.pth")
    print("✅ 双加权损失模型已保存：dae_dual_loss_model.pth")
    return train_losses, val_losses


# ===================== 训练函数2：仅回归损失训练 =====================
def train_pred_only():
    print("\n========== 开始训练：仅回归损失（无重构） ==========")
    model_pred = DAE_SoftSensor(input_dim=35, latent_dim=16).to(device)
    optimizer_pred = optim.Adam(model_pred.parameters(), lr=0.001)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model_pred.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_noisy = model_pred.add_noise(batch_x)
            _, y_pred = model_pred(batch_x_noisy)  # 只取预测值

            # 只计算回归损失，无重构损失
            loss = criterion_pred(y_pred, batch_y)

            optimizer_pred.zero_grad()
            loss.backward()
            optimizer_pred.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model_pred.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                _, y_pred = model_pred(batch_x)
                loss = criterion_pred(y_pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[单损失] Epoch {epoch + 1}/{epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

    # 保存模型
    torch.save(model_pred.state_dict(), "dae_pred_only_model.pth")
    print("✅ 仅回归损失模型已保存：dae_pred_only_model.pth")
    return train_losses, val_losses


# 执行两个训练
train_dual_loss()
train_pred_only()


# ===================== 6. 测试阶段：噪声鲁棒性对比 =====================
def test_and_plot():
    print("\n========== 开始测试：无噪/低噪/高噪对比 ==========")
    # 加载两个训练好的模型
    model_dual = DAE_SoftSensor(input_dim=35, latent_dim=16).to(device)
    model_dual.load_state_dict(torch.load("dae_dual_loss_model.pth"))
    model_dual.eval()

    model_pred = DAE_SoftSensor(input_dim=35, latent_dim=16).to(device)
    model_pred.load_state_dict(torch.load("dae_pred_only_model.pth"))
    model_pred.eval()

    # 噪声等级
    noise_levels = {
        "无噪声": 0.0,
        "低噪声": 0.1,
        "高噪声": 0.25
    }

    # 存储结果
    y_true_all = []
    pred_dual_all = {"无噪声": [], "低噪声": [], "高噪声": []}
    pred_pred_all = {"无噪声": [], "低噪声": [], "高噪声": []}
    mse_dual = {}
    mse_pred = {}

    # 收集测试集真实值
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            y_true_all.extend(batch_y.cpu().numpy())
    y_true_all = np.array(y_true_all).flatten()

    # 对每种噪声测试
    for name, factor in noise_levels.items():
        y_pred_dual = []
        y_pred_pred = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                # 加噪声
                if factor > 0:
                    batch_x = model_dual.add_noise(batch_x, noise_factor=factor)
                # 双损失模型
                _, pred_d = model_dual(batch_x)
                y_pred_dual.extend(pred_d.cpu().numpy())
                # 单损失模型
                _, pred_p = model_pred(batch_x)
                y_pred_pred.extend(pred_p.cpu().numpy())

        y_pred_dual = np.array(y_pred_dual).flatten()
        y_pred_pred = np.array(y_pred_pred).flatten()

        pred_dual_all[name] = y_pred_dual
        pred_pred_all[name] = y_pred_pred

        # 计算MSE
        mse_dual[name] = np.mean((y_true_all - y_pred_dual) ** 2)
        mse_pred[name] = np.mean((y_true_all - y_pred_pred) ** 2)

        print(f"\n--- {name} ---")
        print(f"双加权损失模型 MSE: {mse_dual[name]:.6f}")
        print(f"仅回归损失模型 MSE: {mse_pred[name]:.6f}")

    # ===================== 绘制3张对比图 =====================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 无噪声
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_all, label='真实值', linewidth=1.5)
    plt.plot(pred_dual_all["无噪声"], label='双损失模型预测', linewidth=1.5)
    plt.plot(pred_pred_all["无噪声"], label='仅回归模型预测', linewidth=1.5)
    plt.title(f"无噪声对比 | 双损失MSE={mse_dual['无噪声']:.4f} | 单损失MSE={mse_pred['无噪声']:.4f}")
    plt.xlabel("样本序号")
    plt.ylabel("预测值")
    plt.legend()
    plt.tight_layout()
    plt.savefig("对比_无噪声.png", dpi=300)
    plt.show()

    # 2. 低噪声
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_all, label='真实值', linewidth=1.5)
    plt.plot(pred_dual_all["低噪声"], label='双损失模型预测', linewidth=1.5)
    plt.plot(pred_pred_all["低噪声"], label='仅回归模型预测', linewidth=1.5)
    plt.title(f"低噪声(0.05)对比 | 双损失MSE={mse_dual['低噪声']:.4f} | 单损失MSE={mse_pred['低噪声']:.4f}")
    plt.xlabel("样本序号")
    plt.ylabel("预测值")
    plt.legend()
    plt.tight_layout()
    plt.savefig("对比_低噪声.png", dpi=300)
    plt.show()

    # 3. 高噪声
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_all, label='真实值', linewidth=1.5)
    plt.plot(pred_dual_all["高噪声"], label='双损失模型预测', linewidth=1.5)
    plt.plot(pred_pred_all["高噪声"], label='仅回归模型预测', linewidth=1.5)
    plt.title(f"高噪声(0.2)对比 | 双损失MSE={mse_dual['高噪声']:.4f} | 单损失MSE={mse_pred['高噪声']:.4f}")
    plt.xlabel("样本序号")
    plt.ylabel("预测值")
    plt.legend()
    plt.tight_layout()
    plt.savefig("对比_高噪声.png", dpi=300)
    plt.show()


# 执行测试与绘图
test_and_plot()