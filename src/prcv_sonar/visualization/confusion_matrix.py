"""
混淆矩阵可视化模块

绘制LDA和SVM的混淆矩阵。
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def plot_confusion_matrices(y_test, y_pred_lda, y_pred_svm, label_encoder, output_dir='output'):
    """
    绘制两个算法的混淆矩阵
    
    参数:
        y_test: 真实标签
        y_pred_lda: LDA预测标签
        y_pred_svm: SVM预测标签
        label_encoder: 标签编码器
        output_dir: 输出目录，默认为'output'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cm_lda = confusion_matrix(y_test, y_pred_lda)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    
    class_names = label_encoder.classes_
    
    sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=axes[0], cbar_kws={'label': '样本数量'})
    axes[0].set_title("Fisher's LDA 混淆矩阵", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('预测标签', fontsize=12)
    axes[0].set_ylabel('真实标签', fontsize=12)
    
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=axes[1], cbar_kws={'label': '样本数量'})
    axes[1].set_title("SVM 混淆矩阵", fontsize=14, fontweight='bold')
    axes[1].set_xlabel('预测标签', fontsize=12)
    axes[1].set_ylabel('真实标签', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.pdf'), bbox_inches='tight')
    print("混淆矩阵图已保存为 confusion_matrices.png 和 confusion_matrices.pdf")
    plt.close()
