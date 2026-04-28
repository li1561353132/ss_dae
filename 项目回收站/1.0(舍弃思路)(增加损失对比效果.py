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


# # ===================== 4. 模型初始化 =====================
model = DAE_SoftSensor(input_dim=35, latent_dim=16).to(device)
# print("模型子模块:"),print(model)
# # print("encoder模块:",model._modules['encoder'])
# print("模型参数：")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

criterion_recon = nn.MSELoss()  # 重构损失
criterion_pred = nn.MSELoss()  # 预测损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 150
alpha = 0.8  # 预测损失权重，重构损失权重=0.2

# ===================== 5. 训练模型 =====================
# ① 封装MSE损失函数（统一调用，论文规范）
def mse_loss(input, target):
    return nn.functional.mse_loss(input, target)

# 记录指标：总损失、重构MSE、回归MSE
train_total_losses = []
train_recon_mse = []
train_pred_mse = []

val_total_losses = []
val_recon_mse = []
val_pred_mse = []

print("\n开始训练...")
for epoch in range(epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0.0
    recon_train_mse = 0.0
    pred_train_mse = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # ② 对X添加噪声（调用类内函数）
        batch_x_noisy = model.add_noise(batch_x)

        # 前向传播
        x_recon, y_pred = model(batch_x_noisy)

        # ② 计算3个损失：重构损失、回归损失、加权总损失
        loss_recon = mse_loss(x_recon, batch_x)
        loss_pred = mse_loss(y_pred, batch_y)
        loss_total = alpha * loss_pred + (1 - alpha) * loss_recon

        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # 累计
        total_train_loss += loss_total.item() * batch_x.size(0)
        recon_train_mse += loss_recon.item() * batch_x.size(0)
        pred_train_mse += loss_pred.item() * batch_x.size(0)

    # ③ 训练集：平均总损失、重构MSE；④ 训练集：回归MSE
    total_train_loss /= len(train_loader.dataset)
    recon_train_mse /= len(train_loader.dataset)
    pred_train_mse /= len(train_loader.dataset)

    train_total_losses.append(total_train_loss)
    train_recon_mse.append(recon_train_mse)
    train_pred_mse.append(pred_train_mse)

    # 验证阶段
    model.eval()
    total_val_loss = 0.0
    recon_val_mse = 0.0
    pred_val_mse = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_recon, y_pred = model(batch_x)

            # 损失计算
            loss_recon = mse_loss(x_recon, batch_x)
            loss_pred = mse_loss(y_pred, batch_y)
            loss_total = alpha * loss_pred + (1 - alpha) * loss_recon

            total_val_loss += loss_total.item() * batch_x.size(0)
            recon_val_mse += loss_recon.item() * batch_x.size(0)
            pred_val_mse += loss_pred.item() * batch_x.size(0)

    # ③ 验证集：重构MSE；④ 验证集：回归MSE
    total_val_loss /= len(val_loader.dataset)
    recon_val_mse /= len(val_loader.dataset)
    pred_val_mse /= len(val_loader.dataset)

    val_total_losses.append(total_val_loss)
    val_recon_mse.append(recon_val_mse)
    val_pred_mse.append(pred_val_mse)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"训练总损失:{total_train_loss:.4f} | 训练重构MSE:{recon_train_mse:.4f} | 训练回归MSE:{pred_train_mse:.4f} | "
              f"验证总损失:{total_val_loss:.4f} | 验证重构MSE:{recon_val_mse:.4f} | 验证回归MSE:{pred_val_mse:.4f}")

# ===================== 6. 模型测试与评估 =====================
model.eval()

# 测试集指标存储
test_recon_noisy_mse = 0.0  # 原X vs 噪声X 的MSE
test_recon_mse = 0.0       # 原X vs 重构X 的MSE
test_pred_mse = 0.0        # 原y vs 预测y 的MSE

y_pred_list = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # ⑤ 测试集：添加噪声
        batch_x_noisy = model.add_noise(batch_x)
        x_recon, y_pred = model(batch_x_noisy)

        # 测试集重构：原数据 vs 噪声数据 MSE
        noisy_mse = mse_loss(batch_x_noisy, batch_x)
        # 测试集重构：原数据 vs 重构数据 MSE
        recon_mse = mse_loss(x_recon, batch_x)
        # 测试集回归 MSE
        pred_mse = mse_loss(y_pred, batch_y)

        test_recon_noisy_mse += noisy_mse.item() * batch_x.size(0)
        test_recon_mse += recon_mse.item() * batch_x.size(0)
        test_pred_mse += pred_mse.item() * batch_x.size(0)

        y_pred_list.append(y_pred.cpu().numpy())

# 平均
test_recon_noisy_mse /= len(test_loader.dataset)
test_recon_mse /= len(test_loader.dataset)
test_pred_mse /= len(test_loader.dataset)

y_pred = np.concatenate(y_pred_list, axis=0)

# 反标准化
y_true = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred)

# 回归评价指标
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# 论文式打印结果
print("\n===== 测试集去噪重构效果对比 =====")
print(f"原始数据 vs 噪声数据 MSE: {test_recon_noisy_mse:.6f}")
print(f"原始数据 vs 重构数据 MSE: {test_recon_mse:.6f}")
print(f"噪声降低比例: {(test_recon_noisy_mse - test_recon_mse) / test_recon_noisy_mse * 100:.2f}%")

print("\n===== 软测量模型评估结果 =====")
print(f"测试集回归MSE: {test_pred_mse:.6f}")
print(f"R² 决定系数: {r2:.4f}")
print(f"RMSE 均方根误差: {rmse:.4f}")
print(f"MAE 平均绝对误差: {mae:.4f}")

# ===================== 7. 绘图可视化（论文规范版 + 新增X去噪对比图） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 先取出一批测试数据用于绘制X对比图
model.eval()
sample_x = []
sample_x_noisy = []
sample_x_recon = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        batch_x_noisy = model.add_noise(batch_x)
        x_recon, _ = model(batch_x_noisy)
        sample_x.append(batch_x.cpu().numpy())
        sample_x_noisy.append(batch_x_noisy.cpu().numpy())
        sample_x_recon.append(x_recon.cpu().numpy())
        break  # 只取第一个batch绘图，避免太密集

# 拼接并取第一个特征维度绘图（代表X整体趋势）
sample_x = np.concatenate(sample_x, axis=0)
sample_x_noisy = np.concatenate(sample_x_noisy, axis=0)
sample_x_recon = np.concatenate(sample_x_recon, axis=0)
# 取第0维特征展示趋势
plot_dim = 0

# 绘制5张子图（总损失、重构MSE、回归MSE、预测y、X原/噪声/重构对比）
plt.figure(figsize=(16, 12))

# 1. 训练/验证总损失
plt.subplot(2, 3, 1)
plt.plot(train_total_losses, label='训练总损失', linewidth=2)
plt.plot(val_total_losses, label='验证总损失', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('训练/验证总损失曲线')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 2. 重构MSE曲线
plt.subplot(2, 3, 2)
plt.plot(train_recon_mse, label='训练重构MSE', linewidth=2)
plt.plot(val_recon_mse, label='验证重构MSE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Reconstruction MSE')
plt.title('训练/验证重构MSE曲线')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 3. 回归MSE曲线
plt.subplot(2, 3, 3)
plt.plot(train_pred_mse, label='训练回归MSE', linewidth=2)
plt.plot(val_pred_mse, label='验证回归MSE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Prediction MSE')
plt.title('训练/验证回归MSE曲线')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 4. 预测值 vs 真实值
plt.subplot(2, 3, 4)
plt.plot(y_true, label='真实BODe', linewidth=1.8)
plt.plot(y_pred, label='预测BODe', linewidth=1.8, alpha=0.8)
plt.xlabel('样本点')
plt.ylabel('BODe浓度')
plt.title(f'软测量预测结果  R²={r2:.4f}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 5. 新增：原数据X、噪声数据、重构数据对比（核心去噪效果图）
plt.subplot(2, 3, 5)
plt.plot(sample_x[:, plot_dim], label='原始X', linewidth=2.5)
plt.plot(sample_x_noisy[:, plot_dim], label='噪声X', linewidth=1.5, alpha=0.7)
plt.plot(sample_x_recon[:, plot_dim], label='重构X', linewidth=2.5)
plt.xlabel('样本点')
plt.ylabel(f'X第{plot_dim}维特征值')
plt.title('原始X / 噪声X / 重构X 对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()