"""
LDA数据分布可视化模块

绘制LDA降维后的数据分布（直方图和KDE）。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_lda_distribution(X_train_lda, y_train, X_test_lda, y_test, label_encoder, output_dir='output'):
    """
    绘制LDA降维后的数据分布（直方图和KDE）
    
    参数:
        X_train_lda: 训练集LDA投影数据
        y_train: 训练集标签
        X_test_lda: 测试集LDA投影数据
        y_test: 测试集标签
        label_encoder: 标签编码器
        output_dir: 输出目录，默认为'output'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    class_names = label_encoder.classes_
    colors = ['#FF6B6B', '#4ECDC4']
    
    for i, class_label in enumerate([0, 1]):
        mask_train = y_train == class_label
        mask_test = y_test == class_label
        
        axes[0].hist(X_train_lda[mask_train], bins=20, alpha=0.6, 
                     label=f'{class_names[class_label]} (训练集)', 
                     color=colors[i], edgecolor='black', linewidth=0.5)
        axes[0].hist(X_test_lda[mask_test], bins=20, alpha=0.4, 
                     label=f'{class_names[class_label]} (测试集)', 
                     color=colors[i], edgecolor='black', linewidth=0.5, hatch='//')
    
    axes[0].set_xlabel('LDA投影值', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title("LDA降维后数据分布（直方图）", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    for i, class_label in enumerate([0, 1]):
        mask_train = y_train == class_label
        mask_test = y_test == class_label
        
        sns.kdeplot(X_train_lda[mask_train].flatten(), ax=axes[1], 
                   label=f'{class_names[class_label]} (训练集)', 
                   color=colors[i], linewidth=2, alpha=0.8)
        sns.kdeplot(X_test_lda[mask_test].flatten(), ax=axes[1], 
                   label=f'{class_names[class_label]} (测试集)', 
                   color=colors[i], linewidth=2, linestyle='--', alpha=0.6)
    
    axes[1].set_xlabel('LDA投影值', fontsize=12)
    axes[1].set_ylabel('概率密度', fontsize=12)
    axes[1].set_title("LDA降维后数据分布（核密度估计）", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lda_distribution.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'lda_distribution.pdf'), bbox_inches='tight')
    print("LDA分布图已保存为 lda_distribution.png 和 lda_distribution.pdf")
    plt.close()
