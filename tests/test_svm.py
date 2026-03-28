"""
测试SVM分类器
"""

import pytest
import numpy as np
from src.prml_sonar.models.svm import SVMClassifier


def test_svm_initialization():
    """
    测试SVM分类器初始化
    """
    svm = SVMClassifier(cv=5, random_state=42)
    assert svm.cv == 5, "cv参数设置不正确"
    assert svm.random_state == 42, "random_state参数设置不正确"
    print("SVM初始化测试通过")


def test_svm_fit_predict():
    """
    测试SVM训练和预测
    """
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(30, 10)
    y_test = np.random.randint(0, 2, 30)
    
    svm = SVMClassifier(cv=3, random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    assert y_pred.shape == y_test.shape, "预测结果形状不正确"
    assert len(np.unique(y_pred)) <= 2, "预测结果类别数量不正确"
    print("SVM训练和预测测试通过")


def test_svm_best_params():
    """
    测试SVM超参数调优
    """
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    svm = SVMClassifier(cv=3, random_state=42)
    svm.fit(X_train, y_train)
    
    best_params = svm.get_best_params()
    assert best_params is not None, "最优参数为空"
    assert 'C' in best_params, "缺少C参数"
    assert 'kernel' in best_params, "缺少kernel参数"
    print("SVM超参数调优测试通过")


def test_svm_score():
    """
    测试SVM评分功能
    """
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(30, 10)
    y_test = np.random.randint(0, 2, 30)
    
    svm = SVMClassifier(cv=3, random_state=42)
    svm.fit(X_train, y_train)
    
    score = svm.score(X_test, y_test)
    assert 0 <= score <= 1, "准确率应该在0到1之间"
    print("SVM评分测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
