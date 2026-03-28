import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
DATA_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def load_data():
    """
    加载Sonar数据集
    优先从本地data文件夹读取，失败则从UCI官方链接下载
    """
    local_path = os.path.join(DATA_DIR, 'sonar.all-data')
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    
    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, header=None)
            print(f"从本地加载数据成功！数据集形状: {df.shape}")
        else:
            df = pd.read_csv(url, header=None)
            df.to_csv(local_path, index=False, header=False)
            print(f"从网络下载数据成功！数据已保存到 {local_path}")
            print(f"数据集形状: {df.shape}")
        
        print(f"特征数量: {df.shape[1] - 1}, 样本数量: {df.shape[0]}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请手动下载数据集并保存到 data/sonar.all-data")
        return None

def preprocess_data(df):
    """
    数据预处理：标签转换、数据划分、标准化
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"标签编码: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("数据标准化完成")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

def train_lda(X_train, y_train, X_test):
    """
    训练Fisher's LDA模型
    """
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_train, y_train)
    
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    
    y_pred_lda = lda.predict(X_test)
    accuracy_lda = accuracy_score(y_test, y_pred_lda)
    print(f"LDA测试集准确率: {accuracy_lda:.4f}")
    
    return lda, X_train_lda, X_test_lda, y_pred_lda, accuracy_lda

def train_svm(X_train, y_train, X_test, y_test):
    """
    训练SVM模型并进行超参数调优
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    print(f"SVM最优参数: {grid_search.best_params_}")
    print(f"SVM交叉验证最佳准确率: {grid_search.best_score_:.4f}")
    
    y_pred_svm = best_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM测试集准确率: {accuracy_svm:.4f}")
    
    return best_svm, y_pred_svm, accuracy_svm, grid_search

def plot_confusion_matrices(y_test, y_pred_lda, y_pred_svm, label_encoder):
    """
    绘制两个算法的混淆矩阵
    """
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.pdf'), bbox_inches='tight')
    print("混淆矩阵图已保存为 confusion_matrices.png 和 confusion_matrices.pdf")
    plt.close()

def plot_lda_distribution(X_train_lda, y_train, X_test_lda, y_test, label_encoder):
    """
    绘制LDA降维后的数据分布（直方图和KDE）
    """
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'lda_distribution.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'lda_distribution.pdf'), bbox_inches='tight')
    print("LDA分布图已保存为 lda_distribution.png 和 lda_distribution.pdf")
    plt.close()

def plot_svm_decision_boundary(X_train, y_train, X_test, y_test, svm_model, label_encoder):
    """
    使用PCA降维至2维并绘制SVM决策边界
    """
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'svm_decision_boundary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'svm_decision_boundary.pdf'), bbox_inches='tight')
    print("SVM决策边界图已保存为 svm_decision_boundary.png 和 svm_decision_boundary.pdf")
    plt.close()

def main():
    """
    主函数：执行完整的机器学习流程
    """
    print("=" * 60)
    print("Sonar数据集机器学习任务 - Fisher's LDA vs SVM")
    print("=" * 60)
    print()
    
    df = load_data()
    if df is None:
        return
    
    print("\n" + "=" * 60)
    print("数据预处理")
    print("=" * 60)
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
    
    print("\n" + "=" * 60)
    print("训练 Fisher's LDA 模型")
    print("=" * 60)
    lda, X_train_lda, X_test_lda, y_pred_lda, accuracy_lda = train_lda(X_train, y_train, X_test)
    
    print("\n" + "=" * 60)
    print("训练 SVM 模型（带超参数调优）")
    print("=" * 60)
    svm, y_pred_svm, accuracy_svm, grid_search = train_svm(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)
    print(f"Fisher's LDA 准确率: {accuracy_lda:.4f}")
    print(f"SVM 准确率: {accuracy_svm:.4f}")
    print()
    
    print("\n" + "=" * 60)
    print("LDA 详细分类报告")
    print("=" * 60)
    print(classification_report(y_test, y_pred_lda, target_names=label_encoder.classes_))
    
    print("\n" + "=" * 60)
    print("SVM 详细分类报告")
    print("=" * 60)
    print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
    
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    plot_confusion_matrices(y_test, y_pred_lda, y_pred_svm, label_encoder)
    plot_lda_distribution(X_train_lda, y_train, X_test_lda, y_test, label_encoder)
    plot_svm_decision_boundary(X_train, y_train, X_test, y_test, svm, label_encoder)
    
    print("\n" + "=" * 60)
    print("任务完成！所有图表已保存")
    print("=" * 60)

if __name__ == "__main__":
    main()
