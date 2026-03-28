"""
可视化模块

包含混淆矩阵、LDA分布图、SVM决策边界图等可视化函数。
"""

from .confusion_matrix import plot_confusion_matrices
from .lda_plot import plot_lda_distribution
from .svm_boundary import plot_svm_decision_boundary

__all__ = ["plot_confusion_matrices", "plot_lda_distribution", "plot_svm_decision_boundary"]
