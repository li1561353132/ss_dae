# ss_dae - 基于去噪自编码器的软测量建模项目
污水处理 BSM1 数据预测 | DAE / AE / BP 神经网络对比实验

---

## 📁 项目文件结构
```text
ss_dae/
├── config/                 # 训练配置文件
│   └── config.yaml
├── data/                   # 数据集目录
│   └── BSM1SUNNY.csv
├── models/                 # 模型定义
│   ├── AE.py               # 标准自编码器
│   ├── BP.py               # BP 神经网络基线模型
│   └── DAE.py              # 去噪自编码器
├── output/                 # 输出结果目录
│   ├── plt_png/            # 可视化绘图结果（损失曲线、预测对比图等）
│   └── trained_Model/      # 训练完成的模型权重（.pth 文件）
├── trainers/                # 训练与测试逻辑
│   ├── tester.py           # 模型测试器
│   └── trainer.py          # 模型训练器
├── utils/                  # 工具函数
│   ├── grid.py             # 超参数网格搜索工具
│   └── load_BSM1SUNNY.py   # BSM1 数据集加载、预处理、标准化与划分
├── main.py                 # 项目主入口（配置加载、训练/测试启动）
├── test.py                 # 快速调试/测试脚本
└── README.md               # 项目说明文档