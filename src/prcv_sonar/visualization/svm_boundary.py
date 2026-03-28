"""
SVM决策边界可视化模块

使用PCA降维至2维并绘制SVM决策边界。
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os


def plot_svm_decision_boundary(X_train, y_train, X_test, y_test, svm_model, 
                               label_encoder, output_dir='output'):
    """
    使用PCA降维至2维并绘制SVM决策边界
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        svm_model: 训练好的SVM模型
        label_encoder: 标签编码器
        output_dir: 输出目录，默认为'output'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    svm_pca = SVC(kernel='rbf', C=svm_model.C, gamma=svm_model.gamma, random_state=42)
    svm_pca.fit(X_train_pca, y_train)
    
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    class_names = label_encoder.classes_
    colors = ['#FF6B6B', '#4ECDC4']
    markers = ['o', 's']
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu_r')
    
    for i, class_label in enumerate([0, 1]):
        mask_train = y_train == class_label
        mask_test = y_test == class_label
        
        ax.scatter(X_train_pca[mask_train, 0], X_train_pca[mask_train, 1],
                  c=colors[i], marker=markers[i], s=80, alpha=0.7,
                  label=f'{class_names[class_label]} (训练集)', edgecolors='black', linewidth=0.5)
        ax.scatter(X_test_pca[mask_test, 0], X_test_pca[mask_test, 1],
                  c=colors[i], marker=markers[i], s=120, alpha=0.9,
                  label=f'{class_names[class_label]} (测试集)', edgecolors='black', linewidth=2)
    
    ax.set_xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    ax.set_ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    ax.set_title(f'SVM决策边界 (PCA降维至2维)\n核函数: RBF, C={svm_model.C}, gamma={svm_model.gamma}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svm_decision_boundary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'svm_decision_boundary.pdf'), bbox_inches='tight')
    print("SVM决策边界图已保存为 svm_decision_boundary.png 和 svm_decision_boundary.pdf")
    plt.close()
