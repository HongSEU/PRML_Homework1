"""
混淆矩阵可视化模块

绘制SVM的混淆矩阵。
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def plot_confusion_matrices(y_test, y_pred_svm, unused_param, label_encoder, output_dir='output'):
    """
    绘制SVM的混淆矩阵
    
    参数:
        y_test: 真实标签
        y_pred_svm: SVM预测标签
        unused_param: 未使用的参数（保持向后兼容）
        label_encoder: 标签编码器
        output_dir: 输出目录，默认为'output'
    """
    # 使用绝对路径
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"使用输出目录: {output_dir}")
    
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    class_names = label_encoder.classes_
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("SVM Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix_svm.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存: {output_path}")
    print(f"文件是否存在: {os.path.exists(output_path)}")
    plt.close()
