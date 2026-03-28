"""
PRCV Sonar Classification Package

一个用于Sonar数据集分类的机器学习包，实现了Fisher's LDA和SVM算法。
"""

__version__ = "0.1.0"
__author__ = "PRCV Course"

from .models.lda import LDAClassifier
from .models.svm import SVMClassifier
from .utils.data_loader import load_sonar_data
from .utils.preprocessor import DataPreprocessor

__all__ = [
    "LDAClassifier",
    "SVMClassifier",
    "load_sonar_data",
    "DataPreprocessor",
]
