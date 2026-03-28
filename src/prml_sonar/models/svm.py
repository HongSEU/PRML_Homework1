"""
支持向量机（SVM）分类器

实现SVM算法，支持超参数调优。
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class SVMClassifier:
    """
    SVM分类器
    
    支持多种核函数和超参数调优。
    """
    
    def __init__(self, param_grid=None, cv=5, random_state=42):
        """
        初始化SVM分类器
        
        参数:
            param_grid: 超参数搜索网格，默认为None使用默认网格
            cv: 交叉验证折数，默认为5
            random_state: 随机种子，默认为42
        """
        self.cv = cv
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_score = None
        
        if param_grid is None:
            self.param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        else:
            self.param_grid = param_grid
        
    def fit(self, X_train, y_train):
        """
        训练SVM模型（带超参数调优）
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        svm = SVC(random_state=self.random_state)
        grid_search = GridSearchCV(
            svm, self.param_grid, cv=self.cv, 
            scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
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
    
    def get_best_params(self):
        """
        获取最优超参数
        
        返回:
            最优参数字典
        """
        return self.best_params
    
    def get_best_score(self):
        """
        获取交叉验证最佳分数
        
        返回:
            最佳准确率
        """
        return self.best_score
