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
    try:
        print("开始绘制SVM决策边界...")
        # 使用绝对路径
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用输出目录: {output_dir}")
        
        print("PCA降维...")
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        print(f"PCA降维完成，形状: {X_train_pca.shape}")
        
        print("训练SVM模型...")
        svm_pca = SVC(kernel='rbf', C=svm_model.C, gamma=svm_model.gamma, random_state=42)
        svm_pca.fit(X_train_pca, y_train)
        print("SVM训练完成")
        
        print("生成网格...")
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        print(f"网格形状: {xx.shape}")
        
        print("预测网格点...")
        Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        print("预测完成")
        
        print("创建图表...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        class_names = label_encoder.classes_
        colors = ['#FF6B6B', '#4ECDC4']
        markers = ['o', 's']
        
        print("绘制决策边界...")
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu_r')
        
        print("绘制散点图...")
        for i, class_label in enumerate([0, 1]):
            mask_train = y_train == class_label
            mask_test = y_test == class_label
            
            ax.scatter(X_train_pca[mask_train, 0], X_train_pca[mask_train, 1],
                      c=colors[i], marker=markers[i], s=80, alpha=0.7,
                      label=f'{class_names[class_label]} (train)', edgecolors='black', linewidth=0.5)
            ax.scatter(X_test_pca[mask_test, 0], X_test_pca[mask_test, 1],
                      c=colors[i], marker=markers[i], s=120, alpha=0.9,
                      label=f'{class_names[class_label]} (test)', edgecolors='black', linewidth=2)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        ax.set_title(f'SVM Decision Boundary (PCA)\nRBF kernel, C={svm_model.C}, gamma={svm_model.gamma}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'svm_decision_boundary.png')
        print(f"保存图表到: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"SVM决策边界图已保存: {output_path}")
        print(f"文件是否存在: {os.path.exists(output_path)}")
        plt.close()
        print("SVM决策边界绘制完成！")
    except Exception as e:
        print(f"绘制SVM决策边界时出错: {e}")
        import traceback
        traceback.print_exc()
