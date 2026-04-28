import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 基础测试类（通用模型）
class ModelTester:
    def __init__(self, model, test_loader, device, scaler_y):
        # 初始化传入固定参数
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.scaler_y = scaler_y

        # 类属性：存储返回值（供后续调用）
        self.y_true = None
        self.y_pred = None
        self.r2 = None
        self.rmse = None
        self.mae = None

    def test(self, noise_factor=0.0):
        """核心测试逻辑（通用版本）"""
        self.model.eval()
        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                if noise_factor > 0:
                    batch_x = batch_x + torch.randn_like(batch_x) * noise_factor

                # 通用模型前向传播
                y_pred = self._forward(batch_x)
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(batch_y.cpu().numpy())

        # 后处理
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)
        self.y_true = self.scaler_y.inverse_transform(y_true)
        self.y_pred = self.scaler_y.inverse_transform(y_pred)

        # 计算指标并赋值给类属性
        self.r2 = r2_score(self.y_true, self.y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        self.mae = mean_absolute_error(self.y_true, self.y_pred)

        return self.y_true, self.y_pred, self.r2, self.rmse, self.mae

    def _forward(self, batch_x):
        """通用前向传播（可被子类重写）"""
        return self.model(batch_x)


# DAE 专用测试类（继承基础类，仅重写前向传播）
class DAETester(ModelTester):
    def _forward(self, batch_x):
        """重写：DAE 特殊前向传播"""
        _, y_pred = self.model(batch_x)
        return y_pred