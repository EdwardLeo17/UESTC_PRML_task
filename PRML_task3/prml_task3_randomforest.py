from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns


def plot_digits_images(X, y, images_per_row=4, images_per_col=4, save_path=None):
    """
    显示手写数字数据集中部分图片，打印在一页上

    - X: 数据集的特征部分（图像像素数据）。
    - y: 数据集的标签部分（数字标签）。
    - images_per_row: 每行显示的图片数量。
    - images_per_col: 每列显示的图片数量。
    - save_path: 可选，保存图片的路径（如保存为 PNG 文件）。
    """
    # 设置网格的大小
    fig, axes = plt.subplots(images_per_col, images_per_row, figsize=(images_per_row, images_per_col))
    axes = axes.ravel()  # 拉平成1D数组方便迭代

    # 确定总显示图片数量
    num_images = images_per_row * images_per_col
    for i in range(num_images):
        # 将图片像素数据变换为8x8二维矩阵
        image = X[i].reshape(8, 8)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {y[i]}', fontsize=8)
        axes[i].axis('off')  # 隐藏坐标轴

    # 调整布局
    plt.tight_layout()

    # 保存图片或显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图片保存至: {save_path}")
    else:
        plt.show()


def plot_grid_search_results(grid_search, save_path=None):
    """
    可视化网格搜索结果

    - grid_search: 已经拟合完成的 GridSearchCV 对象。
    - save_path: 可选，保存图片的路径（如保存为 PNG 文件）。
    """
    # 获取结果
    results = pd.DataFrame(grid_search.cv_results_)
    params = results['param_n_estimators'].astype(str) + '-' + results['param_max_depth'].astype(str)
    scores = results['mean_test_score']

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.bar(params, scores, color='skyblue')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("参数组合 (n_estimators-max_depth)")
    plt.ylabel("交叉验证平均分数")
    plt.title("网格搜索结果")
    plt.tight_layout()

    # 保存或展示
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"网格搜索结果保存至: {save_path}")
    else:
        plt.show()


def plot_grid_search_heatmap(grid_search, criterion, save_path=None):
    """
    绘制网格搜索结果的热图，显示不同参数组合的平均测试分数

    - grid_search: 已经拟合完成的 GridSearchCV 对象。
    - criterion: 当前搜索的准则 ('gini' 或 'entropy')。
    - save_path: 可选，保存图片路径（如保存为 PNG 文件）。
    """
    # 获取结果并过滤当前 criterion 的数据
    results_df = pd.DataFrame(grid_search.cv_results_)
    filtered_results = results_df[results_df['param_criterion'] == criterion]

    # 提取参数和分数并创建透视表
    heatmap_data = filtered_results.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_n_estimators'
    )

    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean Test Score'})
    plt.title(f"Grid Search Results ({criterion.capitalize()})")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Max Depth")
    plt.tight_layout()

    # 保存或展示图片
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"热图保存至: {save_path}")
    else:
        plt.show()


def plot_classification_results(X_test, y_test, y_pred, images_per_row=4, images_per_col=4, save_path=None):
    """
    可视化分类结果

    - X_test: 测试集特征。
    - y_test: 测试集真实标签。
    - y_pred: 测试集预测标签。
    - images_per_row: 每行显示图片数量。
    - images_per_col: 每列显示图片数量。
    - save_path: 可选，保存图片路径（如保存为 PNG 文件）。
    """
    fig, axes = plt.subplots(images_per_col, images_per_row, figsize=(images_per_row * 2, images_per_col * 2))
    axes = axes.ravel()
    num_images = images_per_row * images_per_col

    for i in range(num_images):
        image = X_test[i].reshape(8, 8)  # 恢复为8x8图像
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {y_test[i]}\nPred: {y_pred[i]}', fontsize=8,
                          color='green' if y_test[i] == y_pred[i] else 'red')
        axes[i].axis('off')

    plt.tight_layout()

    # 保存或展示
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"分类结果保存至: {save_path}")
    else:
        plt.show()


"""Part1 数据集"""
# 数据集加载
digits = load_digits()  # dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])

n_samples, h, w = digits.images.shape  # 样本个数、高、宽
X = digits.data  # 样本集 (1288, 1850) ***
y = digits.target  # 正确类别 (1288,) ***
target_names = digits.target_names  # 标签名称
n_features = X.shape[1]  # 特征维度1850
n_classes = target_names.shape[0]  # 类别个数

# print(digits.data.shape, type(digits.data))  # 每张图片8*8，一共有1797张图片 ndarray(1797, 64)
# print(digits.target.shape, type(digits.target))  # 每张图片的标签  ndarray(1797,)
# print(digits.feature_names)  # 特征名称
# print(digits.target_names)  # 标签名称 / 类别名称
# print(digits.images.shape, type(digits.images))  # 同样是数据，和data不一样的是data进行了拉伸操作，这个保留了图片尺寸8*8 ndarray(1797, 8, 8)

# 数据集可视化
plot_digits_images(digits.data, digits.target, 10, 10, 'dataset.png')

# 划分训练集 和 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)  # (1437, 64) (360, 64) (1437,) (360,)

"""Part2 模型训练"""
# 使用Gini指数构建随机森林
rf_gini = RandomForestClassifier(criterion='gini', random_state=42)
rf_gini.fit(X_train, y_train)
y_pred_gini = rf_gini.predict(X_test)

# 使用信息增益（entropy）构建随机森林
rf_entropy = RandomForestClassifier(criterion='entropy', random_state=42)
rf_entropy.fit(X_train, y_train)
y_pred_entropy = rf_entropy.predict(X_test)

# 分类性能评估
print("Gini模型准确率：", accuracy_score(y_test, y_pred_gini))
print("信息增益模型准确率：", accuracy_score(y_test, y_pred_entropy))
print("\nGini分类报告：\n", classification_report(y_test, y_pred_gini))
print("\nEntropy分类报告：\n", classification_report(y_test, y_pred_entropy))

"""Part3 超参数调整"""
param_grid = {
    'n_estimators': [10, 50, 100, 150],
    'max_depth': [None, 10, 20, 30, 40],
    'criterion': ['gini', 'entropy']
}
# 网格搜索
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
# 输出最佳参数
print("最佳参数：", grid_search.best_params_)
print("最佳性能：", grid_search.best_score_)

# 分别绘制 Gini 和 Entropy 的网格搜索热图
plot_grid_search_heatmap(grid_search, criterion='gini', save_path="./grid_search_gini.png")
plot_grid_search_heatmap(grid_search, criterion='entropy', save_path="./grid_search_entropy.png")

# 分类结果可视化
plot_classification_results(X_test, y_test, y_pred_gini, images_per_row=4, images_per_col=4, save_path="./classification_results.png")