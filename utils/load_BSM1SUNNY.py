import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class bsm1sunny_data:
    def __init__(self, csv_path="BSM1SUNNY.csv", batch_size=32):
        # 初始化参数
        self.csv_path = csv_path
        self.batch_size = batch_size

        # 后续需要的所有变量
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.scaler_X, self.scaler_y = None, None

    def split_dataset(self, X, y):
        """
        【独立封装的数据集划分函数】
        时序数据按7:1:2划分训练集、验证集、测试集（不打乱）
        返回：X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw
        """
        total_len = len(X)
        size_train = int(0.7 * total_len)
        size_val = int(0.1 * total_len)
        size_test = total_len - size_train - size_val

        X_train_raw, y_train_raw = X[:size_train], y[:size_train]
        X_val_raw, y_val_raw = X[size_train:size_train + size_val], y[size_train:size_train + size_val]
        X_test_raw, y_test_raw = X[-size_test:], y[-size_test:]

        return X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw

    def load_bsm1sunny_data(self):
        """
        加载BSM1SUNNY.csv数据，完成预处理、划分、标准化、生成DataLoader
        加载变量：X_train,y_train,......,train_loader, val_loader, test_loader, scaler_X, scaler_y
        """
        # 1. 读取数据
        df = pd.read_csv(self.csv_path)      # <class 'pandas.core.frame.DataFrame'>

        # 2. 划分 特征X 和 标签y（最后1列）转列向量             <class 'numpy.ndarray'>
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        # print(X.shape, y.shape)

        # 3、时序数据划分数据集（不打乱）  训练集 验证集 测试集    <class 'numpy.ndarray'>
        # X,y   -->    X_train_raw, y_train_raw 、 X_val_raw, y_val_raw 、 X_test_raw, y_test_raw
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = self.split_dataset(X, y)

        # 4、标准化         ._raw --> .                     <class 'numpy.ndarray'>
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # 只在训练集上 fit
        self.scaler_X.fit(X_train_raw)
        self.scaler_y.fit(y_train_raw)
        # 分别标准化（训练/验证/测试都用训练集的均值方差）
        self.X_train = self.scaler_X.transform(X_train_raw)
        self.y_train = self.scaler_y.transform(y_train_raw)
        self.X_val = self.scaler_X.transform(X_val_raw)
        self.y_val = self.scaler_y.transform(y_val_raw)
        self.X_test = self.scaler_X.transform(X_test_raw)
        self.y_test = self.scaler_y.transform(y_test_raw)

        # 5. 转为 PyTorch Tensor & DataLoader
        # X_train, y_train --train_dataset-->  train_loader ; X_val, y_val --val_dataset-->  val_loader ; X_test, y_test --test_dataset-->  test_loader

        # 构建数据集     # <class 'torch.utils.data.dataset.TensorDataset'>
        train_dataset = TensorDataset(torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train))
        val_dataset = TensorDataset(torch.FloatTensor(self.X_val), torch.FloatTensor(self.y_val))
        test_dataset = TensorDataset(torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test))
        # print(f"TensorDataset数据:{len(train_dataset), len(val_dataset), len(test_dataset)}")           # TensorDataset数据:(940, 134, 270)

        # 构建加载器     # <class 'torch.utils.data.dataloader.DataLoader'>
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        # print(f"DataLoader数据:{len(train_loader), len(val_loader), len(test_loader)}")                   # DataLoader数据:(30, 5, 9)

        # ===================== 打印数据集信息 =====================
        print("=" * 50)
        print(f"原始数据形状: {df.shape}")
        print(f"训练集: {self.X_train.shape}, {self.y_train.shape}")
        print(f"验证集: {self.X_val.shape}, {self.y_val.shape}")
        print(f"测试集: {self.X_test.shape}, {self.y_test.shape}")
        print(f"DataLoader数量: 训练{len(self.train_loader)} | 验证{len(self.val_loader)} | 测试{len(self.test_loader)}")
        print("=" * 50)

