"""
测试LDA分类器
"""

import pytest
import numpy as np
from src.prcv_sonar.models.lda import LDAClassifier


def test_lda_initialization():
    """
    测试LDA分类器初始化
    """
    lda = LDAClassifier(n_components=1)
    assert lda.n_components == 1, "n_components设置不正确"
    print("LDA初始化测试通过")


def test_lda_fit_predict():
    """
    测试LDA训练和预测
    """
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(30, 10)
    y_test = np.random.randint(0, 2, 30)
    
    lda = LDAClassifier(n_components=1)
    lda.fit(X_train, y_train)
    
    y_pred = lda.predict(X_test)
    assert y_pred.shape == y_test.shape, "预测结果形状不正确"
    assert len(np.unique(y_pred)) <= 2, "预测结果类别数量不正确"
    print("LDA训练和预测测试通过")


def test_lda_transform():
    """
    测试LDA数据转换
    """
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    lda = LDAClassifier(n_components=1)
    lda.fit(X_train, y_train)
    
    X_transformed = lda.transform(X_train)
    assert X_transformed.shape[1] == 1, "转换后维度不正确"
    print("LDA数据转换测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
