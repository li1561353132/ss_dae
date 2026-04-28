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
import yaml

from utils.load_BSM1SUNNY import bsm1sunny_data
from models.DAE import DAE_SoftSensor
from models.AE import AE_OnlyReg
from models.BP import BP_Net
from trainers.trainer import *
from trainers.tester import *
from utils.grid import *


def set_seed(seed=41):
    import random
    import numpy as np
    import torch
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(41) # todo 设置随机种子，可选

# ===================== 2. 数据加载与预处理 =====================
data_path= "data/BSM1SUNNY.csv"
bsm1sunny_data = bsm1sunny_data(csv_path=data_path,batch_size=32)
bsm1sunny_data.load_bsm1sunny_data()
X_train, y_train = bsm1sunny_data.X_train, bsm1sunny_data.y_train
X_val, y_val = bsm1sunny_data.X_val, bsm1sunny_data.y_val
X_test, y_test = bsm1sunny_data.X_test, bsm1sunny_data.y_test
train_loader, val_loader, test_loader = bsm1sunny_data.train_loader, bsm1sunny_data.val_loader, bsm1sunny_data.test_loader
scaler_X, scaler_y = bsm1sunny_data.scaler_X, bsm1sunny_data.scaler_y

# ===================== 4. 模型初始化 =====================
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 读取参数配置文件
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
# 提取训练参数
lr = cfg["training"]["lr"]
epochs = cfg["training"]["epochs"]
alpha_reconloss = cfg["training"]["alpha_reconloss"]
criterion = nn.MSELoss()  # 重构损失
# 提取模型参数
input_dim = cfg["model"]["input_dim"]
latent_dim = cfg["model"]["latent_dim"]

dae_model = DAE_SoftSensor(input_dim=input_dim, latent_dim=latent_dim).to(device)
ae_model = AE_OnlyReg(input_dim=input_dim, latent_dim=latent_dim).to(device)
bp_model = BP_Net(input_dim=input_dim).to(device)
# 初始化优化器 模型参数传入优化器
dae_optimizer = optim.Adam(dae_model.parameters(), lr=lr)
ae_optimizer = optim.Adam(ae_model.parameters(), lr=lr)
bp_optimizer = optim.Adam(bp_model.parameters(), lr=lr)

# ===================== 5. 训练模型 =====================
# 初始化训练器
dae_trainer = DAETrainer(
    model=dae_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=dae_optimizer,
    alpha=alpha_reconloss,
    device=device
)
ae_trainer = ModelTrainer(
    model=ae_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=ae_optimizer,
    device=device
)
bp_trainer = ModelTrainer(
    model=bp_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=bp_optimizer,
    device=device
)
# 开始训练
dae_trainer.train(epochs=epochs)
ae_trainer.train(epochs=epochs)
bp_trainer.train(epochs=epochs)

# 保存dae_model.pth
save_path = "output/trained_Model/dae_model.pth" # todo 保存pth
torch.save(dae_model.state_dict(),save_path)
# 保存dae_model.onnx
dummy_input = torch.randn(1, 35).to(device)
torch.onnx.export(
    dae_model,
    dummy_input,
    "output/trained_Model/dae_model.onnx",    # todo 保存onnx
    opset_version=11
)

# ✅ 训练后拿参数
train_losses = dae_trainer.train_losses
val_losses = dae_trainer.val_losses
dae_model = dae_trainer.model

# ===================== 6. 模型测试与评估 =====================
# 初始化训练器
dae_tester = DAETester(
    model=dae_model,
    test_loader=test_loader,
    scaler_y=scaler_y,  # 你的反标准化器
    device=device
)
ae_tester = ModelTester(
    model=ae_model,
    test_loader=test_loader,
    scaler_y=scaler_y,  # 你的反标准化器
    device=device
)
dp_tester = ModelTester(
    model=bp_model,
    test_loader=test_loader,
    scaler_y=scaler_y,  # 你的反标准化器
    device=device
)

# 设置噪声水平
noise_levels = {
    "无噪声": 0.0,
    "低噪声": 0.05,
    "高噪声": 0.15
}

# 存储所有结果
results = {}
y_true_global = None

# 批量测试
for name, noise in noise_levels.items():
    print(f"\n{'='*60}")
    print(f"               {name} 测试 (noise={noise})")
    print(f"{'='*60}")

    yt, dae_p, dae_r2, dae_rmse, dae_mae = dae_tester.test(noise_factor=noise)
    _, ae_p, ae_r2, ae_rmse, ae_mae = ae_tester.test(noise_factor=noise)
    _, bp_p, bp_r2, bp_rmse, bp_mae = dp_tester.test(noise_factor=noise)

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

# ===================== 7. 绘图可视化 =====================
# 损失曲线
plot_loss_curve(dae_trainer.train_losses,dae_trainer.val_losses,save_path="output/plt_png/训练损失曲线")

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
plt.savefig("output/plt_png/无噪声环境预测图.png", dpi=300, bbox_inches='tight')    # todo 保存png

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
plt.savefig("output/plt_png/低噪声环境预测图.png", dpi=300, bbox_inches='tight')    # todo 保存png

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
plt.savefig("output/plt_png/高噪声环境预测图.png", dpi=300, bbox_inches='tight')    # todo 保存png

plt.show()