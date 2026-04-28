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
# print("encoder模块:"model._modules['encoder'])
# print("模型参数：")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

criterion_recon = nn.MSELoss()  # 重构损失
criterion_pred = nn.MSELoss()  # 预测损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 150
alpha = 0.8  # 预测损失权重，重构损失权重=0.2

# # ===================== 5. 训练模型 =====================
# train_losses = []
# val_losses = []
#
# print("\n开始训练...")
# for epoch in range(epochs):
#     # 训练阶段
#     model.train()
#     train_loss = 0.0
#     for batch_x, batch_y in train_loader:
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)
#
#         # 加噪声
#         batch_x_noisy = model.add_noise(batch_x)
#
#         # 前向传播
#         x_recon, y_pred = model(batch_x_noisy)
#
#         # 计算损失
#         loss_recon = criterion_recon(x_recon, batch_x)
#         loss_pred = criterion_pred(y_pred, batch_y)
#         loss = alpha * loss_pred + (1 - alpha) * loss_recon
#
#         # 反向传播
#         optimizer.zero_grad() # 梯度置零 清除上次迭代的梯度
#         loss.backward() # 计算损失函数的梯度
#         optimizer.step() # 更新模型参数
#
#         train_loss += loss.item() * batch_x.size(0)
#
#     train_loss /= len(train_loader.dataset)
#     train_losses.append(train_loss)
#
#     # 验证阶段
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch_x, batch_y in val_loader:
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             x_recon, y_pred = model(batch_x)
#             loss_recon = criterion_recon(x_recon, batch_x)
#             loss_pred = criterion_pred(y_pred, batch_y)
#             loss = alpha * loss_pred + (1 - alpha) * loss_recon
#             val_loss += loss.item() * batch_x.size(0)
#
#     val_loss /= len(val_loader.dataset)
#     val_losses.append(val_loss)
#
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch + 1}/{epochs}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
#
# # ===================== 6. 模型测试与评估 =====================
# model.eval()
# y_pred_list = []
# with torch.no_grad():
#     for batch_x, _ in test_loader:
#         batch_x = batch_x.to(device)
#         _, y_pred = model(batch_x)
#         y_pred_list.append(y_pred.cpu().numpy())
#
# y_pred = np.concatenate(y_pred_list, axis=0)
#
# # 反标准化（还原真实值）
# y_true = scaler_y.inverse_transform(y_test)
# y_pred = scaler_y.inverse_transform(y_pred)
#
# # 计算评价指标
# r2 = r2_score(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# mae = mean_absolute_error(y_true, y_pred)
#
# print("\n===== 软测量模型评估结果 =====")
# print(f"R² 决定系数: {r2:.4f}")
# print(f"RMSE 均方根误差: {rmse:.4f}")
# print(f"MAE 平均绝对误差: {mae:.4f}")
#
# # ===================== 7. 绘图可视化 =====================
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 损失曲线
# plt.figure()  # 图1
# plt.plot(train_losses, label='训练损失', linewidth=2)
# plt.plot(val_losses, label='验证损失', linewidth=2)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('训练/验证损失曲线')
# plt.legend()
# plt.grid(True)
#
# # 预测值vs真实值
# plt.figure()  # 图2
# plt.plot(y_true, label='真实BODe', linewidth=1.5)
# plt.plot(y_pred, label='预测BODe', linewidth=1.5, alpha=0.8)
# plt.xlabel('样本点')
# plt.ylabel('BODe浓度')
# plt.title(f'软测量预测结果  R²={r2:.4f}')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
#
# plt.show()