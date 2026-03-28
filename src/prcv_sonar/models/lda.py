"""
Fisher's 线性判别分析（LDA）分类器

实现Fisher's LDA算法，用于二分类任务。
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report


class LDAClassifier:
    """
    Fisher's LDA分类器
    
    将高维数据降维到1维，寻找最优投影方向。
    """
    
    def __init__(self, n_components=1):
        """
        初始化LDA分类器
        
        参数:
            n_components: 降维后的维度，默认为1
        """
        self.n_components = n_components
        self.model = LinearDiscriminantAnalysis(n_components=n_components)
        self.X_train_lda = None
        self.X_test_lda = None
        
    def fit(self, X_train, y_train):
        """
        训练LDA模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        self.model.fit(X_train, y_train)
        self.X_train_lda = self.model.transform(X_train)
        
    def transform(self, X):
        """
        将数据转换到LDA空间
        
        参数:
            X: 输入数据
            
        返回:
            转换后的数据
        """
        return self.model.transform(X)
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 输入数据
            
        返回:
            预测的类别标签
        """
        return self.model.predict(X)
    
    def score(self, X_test, y_test):
        """
        计算测试集准确率
        
        参数:
            X_test: 测试集特征
            y_test: 测试集标签
            
        返回:
            准确率
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def get_classification_report(self, X_test, y_test, target_names=None):
        """
        获取详细的分类报告
        
        参数:
            X_test: 测试集特征
            y_test: 测试集标签
            target_names: 类别名称列表
            
        返回:
            分类报告字符串
        """
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred, target_names=target_names)
    
    def get_coefficients(self):
        """
        获取LDA系数（投影方向）
        
        返回:
            LDA系数向量
        """
        return self.model.coef_
