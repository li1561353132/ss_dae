import torch

class ModelTrainer:
    """训练器：包含 单epoch训练 + 完整训练 + 验证"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        # 初始化就把所有需要的参数传入，后续方法直接用
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # 训练过程数据（自动保存，后续直接用！）
        self.train_losses = []
        self.val_losses = []

    def train_loop(self):
        """单 epoch 训练（你要的独立封装）"""
        self.model.train()
        total_loss = 0.0
        total_samples = len(self.train_loader.dataset)

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向
            y_pred = self.model(batch_x)
            # 损失
            loss = self.criterion(y_pred, batch_y)
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        return total_loss / total_samples

    def validate_loop(self):
        """验证（训练内部使用）"""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                y_pred = self.model(batch_x)
                val_loss += self.criterion(y_pred, batch_y).item() * batch_x.size(0)
        return val_loss / len(self.val_loader.dataset)

    def train(self, epochs):
        """完整训练循环"""
        print("\n开始训练...")
        for epoch in range(epochs):
            train_loss = self.train_loop()
            val_loss = self.validate_loop()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        # 训练完直接返回自身，方便链式调用
        return self


# 子类：DAE训练器（只重写 train_loop）
class DAETrainer(ModelTrainer):  # 继承 ModelTrainer
    """带噪声的DAE训练器：仅重写训练逻辑"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, alpha, device):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device)
        self.alpha = alpha # 子类自己加 alpha 属性 ✅

    def train_loop(self):
        """单 epoch 训练（重写：加噪声版本）"""
        self.model.train()
        total_loss = 0.0
        total_samples = len(self.train_loader.dataset)

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 加噪声（只有这里不一样！）
            batch_x_noisy = self.model.add_noise(batch_x)
            # 前向
            x_recon, y_pred = self.model(batch_x_noisy)
            # 损失
            loss_recon = self.criterion(x_recon, batch_x)
            loss_pred = self.criterion(y_pred, batch_y)
            loss = 1.0 * loss_pred + self.alpha * loss_recon # todo 可修改训练器损失函数，alpha：dae双损失参数
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        return total_loss / total_samples

    def validate_loop(self):
        """验证（训练内部使用）"""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                x_recon, y_pred = self.model(batch_x)
                val_loss += self.criterion(y_pred, batch_y).item() * batch_x.size(0)
        return val_loss / len(self.val_loader.dataset)

