"""
数据预处理模块

负责标签编码、数据划分、标准化等预处理操作。
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """
    数据预处理器
    
    处理标签编码、数据划分和标准化。
    """
    
    def __init__(self, test_size=0.3, random_state=42):
        """
        初始化数据预处理器
        
        参数:
            test_size: 测试集比例，默认为0.3
            random_state: 随机种子，默认为42
        """
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def fit_transform(self, df):
        """
        拟合并转换数据
        
        参数:
            df: 原始数据框
            
        返回:
            X_train_scaled: 标准化后的训练集特征
            X_test_scaled: 标准化后的测试集特征
            y_train: 训练集标签
            y_test: 测试集标签
        """
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"标签编码: {dict(zip(self.label_encoder.classes_, 
                                   self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, 
            random_state=self.random_state, stratify=y_encoded
        )
        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("数据标准化完成")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def transform(self, X):
        """
        使用已拟合的scaler转换新数据
        
        参数:
            X: 输入数据
            
        返回:
            标准化后的数据
        """
        return self.scaler.transform(X)
    
    def get_label_encoder(self):
        """
        获取标签编码器
        
        返回:
            LabelEncoder对象
        """
        return self.label_encoder
    
    def get_class_names(self):
        """
        获取类别名称
        
        返回:
            类别名称列表
        """
        return self.label_encoder.classes_
