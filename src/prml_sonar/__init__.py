"""
PRCV Sonar Classification Package

一个用于Sonar数据集分类的机器学习包，实现了SVM算法。
"""

__version__ = "0.1.0"
__author__ = "PRCV Course"

from .models.svm import SVMClassifier
from .utils.data_loader import load_sonar_data
from .utils.preprocessor import DataPreprocessor

__all__ = [
    "SVMClassifier",
    "load_sonar_data",
    "DataPreprocessor",
]
