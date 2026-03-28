"""
测试数据加载功能
"""

import pytest
import pandas as pd
import os
from src.prml_sonar.utils.data_loader import load_sonar_data


def test_load_sonar_data():
    """
    测试数据加载功能
    """
    df = load_sonar_data()
    
    assert df is not None, "数据加载失败"
    assert isinstance(df, pd.DataFrame), "返回的不是DataFrame"
    assert df.shape[0] > 0, "数据集为空"
    assert df.shape[1] == 61, f"特征数量不正确，期望61，实际{df.shape[1]}"
    print("数据加载测试通过")


def test_data_columns():
    """
    测试数据列结构
    """
    df = load_sonar_data()
    
    assert df.iloc[:, -1].nunique() == 2, "标签类别数量不正确"
    assert df.iloc[:, :-1].dtypes.all() == float, "特征列应该为浮点数"
    print("数据列结构测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
