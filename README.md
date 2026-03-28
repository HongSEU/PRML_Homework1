# Sonar数据集分类任务 - Fisher's LDA vs SVM

## 项目简介

本项目在"Sonar, Mines vs. Rocks"数据集上实现并比较两种经典机器学习算法：
- Fisher's 线性判别分析（LDA）
- 支持向量机（SVM）

## 项目结构

```
PRCV_Homework1/
├── data/                      # 数据文件夹
│   └── sonar.all-data        # Sonar数据集（需下载）
├── output/                    # 输出文件夹（可视化图表）
│   ├── confusion_matrices.png
│   ├── confusion_matrices.pdf
│   ├── lda_distribution.png
│   ├── lda_distribution.pdf
│   ├── svm_decision_boundary.png
│   └── svm_decision_boundary.pdf
├── models/                    # 模型文件夹（保存训练好的模型）
├── logs/                      # 日志文件夹
├── sonar_classification.py    # 主程序文件
├── requirements.txt           # Python依赖包列表
├── conda_environment_setup.md # Conda环境配置指南
└── README.md                  # 项目说明文档
```

## 环境配置

### 使用 Conda（推荐）

请参考 [conda_environment_setup.md](conda_environment_setup.md) 文件进行环境配置。

### 使用 pip

如果已安装Python环境，可以直接使用pip安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

### 选项1：自动下载（推荐）

运行主程序时会自动从UCI官方链接下载数据集。

### 选项2：手动下载

如果网络连接不稳定，可以手动下载数据集：

1. 访问 UCI Sonar数据集链接：
   https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data

2. 将下载的文件保存到 `data/` 文件夹中，命名为 `sonar.all-data`

## 运行程序

```bash
python sonar_classification.py
```

## 程序输出

### 控制台输出

程序会输出以下信息：
- 数据集基本信息
- 数据预处理过程
- 模型训练过程
- 超参数调优结果
- 模型性能对比
- 详细的分类报告

### 可视化图表

程序会在 `output/` 文件夹中生成以下图表（PNG和PDF格式，300dpi）：

1. **混淆矩阵图** (`confusion_matrices.png/pdf`)
   - Fisher's LDA的混淆矩阵
   - SVM的混淆矩阵

2. **LDA数据分布图** (`lda_distribution.png/pdf`)
   - 直方图展示两类样本在LDA投影方向上的分布
   - 核密度估计（KDE）图展示数据分布

3. **SVM决策边界图** (`svm_decision_boundary.png/pdf`)
   - PCA降维至2维后的数据散点图
   - SVM在2维子空间上的决策边界

## 算法说明

### Fisher's LDA

- 将60维数据降维到1维
- 寻找最优投影方向，最大化类间离散度与类内离散度的比值
- 适用于线性可分的数据

### SVM

- 使用支持向量机构建分类器
- 通过GridSearchCV自动调优超参数：
  - 惩罚系数 C: [0.1, 1, 10, 100]
  - 核函数: ['linear', 'rbf', 'poly']
  - gamma参数: ['scale', 'auto']
- 5折交叉验证寻找最优参数组合

## 性能指标

程序会输出以下性能指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-score）

## 注意事项

1. 确保已安装所有必需的Python包
2. 如果从网络下载数据失败，请手动下载并保存到 `data/` 文件夹
3. 所有图表保存为高分辨率（300dpi），适合插入LaTeX文档
4. 程序会自动创建必要的文件夹结构

## 作者

PRCV课程作业 - 模式识别与计算机视觉

## 参考资料

- UCI Machine Learning Repository: Sonar, Mines vs. Rocks Dataset
- scikit-learn官方文档
- Fisher, R. A. (1936). "The Use of Multiple Measurements in Taxonomic Problems"
