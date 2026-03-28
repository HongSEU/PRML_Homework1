# 模式识别与机器学习 - SVM

## 项目简介

本项目是东南大学未来技术学院《模式识别与机器学习》课程的作业。
实现了在"Sonar, Mines vs. Rocks"数据集上利用支持向量机（SVM）算法，用于分类任务。

## 项目结构

```
PRML_Homework1/
├── data/                          # 数据文件夹
│   └── .gitkeep                  # Git占位文件
├── docs/                          # 文档文件夹
│   └── history.md                # 项目历史记录
├── logs/                          # 日志文件夹
│   └── .gitkeep                  # Git占位文件
├── models/                        # 模型保存文件夹
│   └── .gitkeep                  # Git占位文件
├── output/                        # 输出文件夹（可视化图表）
│   └── .gitkeep                  # Git占位文件
├── src/                           # 源代码文件夹
│   └── prml_sonar/               # 主包
│       ├── __init__.py           # 包初始化
│       ├── main.py               # 主程序入口
│       ├── models/               # 模型模块
│       │   ├── __init__.py
│       │   └── svm.py            # SVM分类器
│       ├── utils/                # 工具模块
│       │   ├── __init__.py
│       │   ├── data_loader.py    # 数据加载
│       │   └── preprocessor.py   # 数据预处理
│       └── visualization/        # 可视化模块
│           ├── __init__.py
│           ├── confusion_matrix.py   # 混淆矩阵
│           └── svm_boundary.py       # SVM决策边界
├── tests/                         # 测试文件夹
│   ├── __init__.py
│   ├── test_data_loader.py       # 数据加载测试
│   └── test_svm.py               # SVM测试
├── .gitignore                     # Git忽略配置
├── pyproject.toml                 # Python项目配置
├── README.md                      # 项目说明文档
├── requirements.txt               # 依赖包列表
└── tox.ini                        # Tox测试配置
```

## 环境配置

### conda环境配置
假设您已经有了conda环境

```bash
conda_env_name=PRML
conda create -n $conda_env_name python=3.9
conda activate $conda_env_name

conda install "numpy>=1.20.0" "pandas>=1.3.0" "scikit-learn>=0.24.0" "matplotlib>=3.3.0" "seaborn>=0.11.0" -c conda-forge -y
```

### 使用 pyproject.toml

```bash
pip install -e .
```

### 开发依赖

```bash
pip install -e ".[dev]"
```

## 运行程序

### 方式1：直接运行主程序

```bash
python -m src.prml_sonar.main
```

或

```bash
cd src
python -m prml_sonar.main
```

### 方式2：作为包运行

```python
from src.prml_sonar.main import main
main()
```

## 数据准备

### 选项1：自动下载（推荐）

运行主程序时会自动从UCI官方链接下载数据集并保存到 `data/` 目录。

### 选项2：手动下载

如果网络连接不稳定，可以手动下载：

1. 访问 UCI Sonar数据集链接：
   https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data

2. 将下载的文件保存到 `data/sonar.all-data`

## 程序输出

### 控制台输出

- 数据集基本信息
- 数据预处理过程
- 模型训练过程
- 超参数调优结果
- 详细的分类报告

### 可视化图表

程序会在 `output/` 文件夹中生成以下图表（PNG格式，300dpi）：

1. **混淆矩阵图** (`confusion_matrix_svm.png`)
   - SVM的混淆矩阵

2. **SVM决策边界图** (`svm_decision_boundary.png`) [可选]
   - PCA降维至2维后的数据散点图
   - SVM在2维子空间上的决策边界

## 模块说明

### models 模块

- **svm.py**: SVM分类器实现
  - `SVMClassifier` 类：封装SVM训练和GridSearchCV超参数调优

### utils 模块

- **data_loader.py**: 数据加载功能
  - `load_sonar_data()`: 从本地或网络加载数据

- **preprocessor.py**: 数据预处理功能
  - `DataPreprocessor` 类：标签编码、数据划分、标准化

### visualization 模块

- **confusion_matrix.py**: 混淆矩阵可视化
- **svm_boundary.py**: SVM决策边界可视化

## 算法说明

### SVM

- 使用支持向量机构建分类器
- 通过GridSearchCV自动调优超参数：
  - 惩罚系数 C: [1, 10]
  - 核函数: ['rbf', 'linear']
  - gamma参数: ['scale']
- 3折交叉验证寻找最优参数组合

## 性能指标

程序会输出以下性能指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-score）

## 测试

运行测试：

```bash
pytest
```

或

```bash
python -m pytest tests/
```

## 代码规范检查

使用tox运行代码检查：

```bash
tox
```

单独运行flake8：

```bash
tox -e flake8
```

单独运行mypy：

```bash
tox -e mypy
```

## 注意事项

1. 确保已安装所有必需的Python包
2. 如果从网络下载数据失败，请手动下载并保存到 `data/` 文件夹
3. 所有图表保存为高分辨率（300dpi），适合插入LaTeX文档
4. 程序会自动创建必要的文件夹结构
5. 项目按照标准Python包规范组织，便于维护和扩展

## 作者

PRML课程作业 - 模式识别与机器学习

## 参考资料

- UCI Machine Learning Repository: Sonar, Mines vs. Rocks Dataset
- scikit-learn官方文档
