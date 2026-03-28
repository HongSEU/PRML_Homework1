"""
数据加载模块

负责从UCI数据集或本地文件加载Sonar数据。
"""

import pandas as pd
import os


def load_sonar_data(data_dir='data'):
    """
    加载Sonar数据集
    
    优先从本地data文件夹读取，失败则从UCI官方链接下载
    
    参数:
        data_dir: 数据目录路径，默认为'data'
        
    返回:
        DataFrame: 包含特征和标签的数据框
        None: 加载失败时返回None
    """
    local_path = os.path.join(data_dir, 'sonar.all-data')
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    
    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, header=None)
            print(f"从本地加载数据成功！数据集形状: {df.shape}")
        else:
            df = pd.read_csv(url, header=None)
            os.makedirs(data_dir, exist_ok=True)
            df.to_csv(local_path, index=False, header=False)
            print(f"从网络下载数据成功！数据已保存到 {local_path}")
            print(f"数据集形状: {df.shape}")
        
        print(f"特征数量: {df.shape[1] - 1}, 样本数量: {df.shape[0]}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请手动下载数据集并保存到 data/sonar.all-data")
        return None
