"""
主程序入口

执行完整的机器学习流程：数据加载、预处理、SVM模型训练、评估和可视化。
"""

import warnings
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.prcv_sonar.models.svm import SVMClassifier
from src.prcv_sonar.utils.data_loader import load_sonar_data
from src.prcv_sonar.utils.preprocessor import DataPreprocessor
from src.prcv_sonar.visualization.confusion_matrix import plot_confusion_matrices
from src.prcv_sonar.visualization.svm_boundary import plot_svm_decision_boundary

warnings.filterwarnings('ignore')

# 暂时禁用中文字体设置，避免字体问题
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False


def main():
    """
    主函数：执行完整的机器学习流程
    """
    print("=" * 60)
    print("Sonar数据集机器学习任务 - SVM")
    print("=" * 60)
    print()
    
    df = load_sonar_data()
    if df is None:
        return
    
    print("\n" + "=" * 60)
    print("数据预处理")
    print("=" * 60)
    preprocessor = DataPreprocessor(test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    label_encoder = preprocessor.get_label_encoder()
    
    print("\n" + "=" * 60)
    print("训练 SVM 模型（带超参数调优）")
    print("=" * 60)
    print("正在进行网格搜索，请稍候...")
    # 减少参数范围以加快运行速度
    param_grid = {
        'C': [1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale']
    }
    svm = SVMClassifier(param_grid=param_grid, cv=3, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = svm.score(X_test, y_test)
    print(f"SVM最优参数: {svm.get_best_params()}")
    print(f"SVM交叉验证最佳准确率: {svm.get_best_score():.4f}")
    print(f"SVM测试集准确率: {accuracy_svm:.4f}")
    
    print("\n" + "=" * 60)
    print("SVM 详细分类报告")
    print("=" * 60)
    print(svm.get_classification_report(X_test, y_test, target_names=label_encoder.classes_))
    
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    print(f"X_train形状: {X_train.shape}")
    print(f"y_train形状: {y_train.shape}")
    print(f"X_test形状: {X_test.shape}")
    print(f"y_test形状: {y_test.shape}")
    # 只绘制SVM的混淆矩阵
    plot_confusion_matrices(y_test, y_pred_svm, y_pred_svm, label_encoder)
    # 暂时跳过SVM决策边界绘制
    print("跳过SVM决策边界绘制")
    
    print("\n" + "=" * 60)
    print("任务完成！所有图表已保存到 output/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
