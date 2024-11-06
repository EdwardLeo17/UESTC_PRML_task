from scipy import io
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# N_COMPONENTS = 5
N_COMPONENTS = 25
KERNEL = 'rbf'
N_FOLDS = 5  # 5折交叉验证


def reshape_images(image_data):
    images = []
    for img in image_data:
        # 将一维数组重塑为19*19的二维数组
        img_reshaped = img.reshape((19, 19))
        img_rotated = np.rot90(img_reshaped, -1)
        images.append(img_rotated)
    return np.array(images)


def print_images(images, labels):
    num_images = images.shape[0]
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))  # 创建一个10x10的子图网格
    axs = axs.flatten()  # 将二维的axs数组展平，方便遍历

    for i in range(num_images):
        # 显示图片
        axs[i].imshow(images[i], cmap='gray')  # 假设图片是灰度图
        axs[i].set_title(f"Label: {labels[i]}")  # 在图片下方标注标签
        axs[i].axis('off')  # 不显示坐标轴

    # 调整子图间距
    plt.tight_layout()
    plt.show()


def display_all_images(image_data, image_label):
    images = reshape_images(image_data)  # (800, 19, 19)
    k = int(len(images) / 100) + 1
    for i in range(k):
        print_images(images[i * 100: (i + 1) * 100], image_label[i * 100: (i + 1) * 100])


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """
    :param images: ndarray.shape=(322, 1850)=(样本个数, 特征个数)
    :param titles: 每张图片的标题
    :param h: 每张图片的尺寸
    :param w: 每张图片的尺寸
    :return: 绘制images，按照3行4列
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(np.rot90(images[i].reshape((h, w)), -1), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    """
    :param y_pred: X_test_pca经过预测后的结果，ndarray.shape=(预测样本个数,)
    :param y_test: 预测样本真实的标签，ndarray.shape=(预测样本个数,)
    :param target_names: 这是一个ndarray,对应了真实的标签名称
    :param i: 第i个预测样本
    :return: 用来生成 第i个预测样本的 下面这种图片标题：
        predicted: face
        true:      face
    """
    y_pred[i] = 0 if y_pred[i] == -1 else y_pred[i]
    y_test[i] = 0 if y_test[i] == -1 else y_test[i]

    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


if __name__ == '__main__':
    """Part1 数据的加载"""
    # 加载数据集
    train_data = io.loadmat('./data/train_data.mat')['train_data']  # (800, 361) 训练集有800个样本，每个19*19
    train_label = io.loadmat('./data/train_label.mat')['train_label'].squeeze()  # (800,) 这800个样本对应的label
    test_data = io.loadmat('./data/test_data.mat')['test_data']  # (420, 361) 有420个用来测试的样本

    target_names = np.array(['non-face', 'face'])
    n_classes = target_names.shape[0]
    h, w = 19, 19

    # 打印图片
    # display_all_images(train_data, train_label)

    # # 统计不同类别的样本数量
    # unique, counts = np.unique(train_label, return_counts=True)
    # print(f"Label: {unique}, Counts: {counts}")  # Label: [-1  1], Counts: [400 400]

    # 划分训练集 和 验证集
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.4,
                                                      random_state=42)  # (640, 361) (160, 361) (640,) (160,)

    """Part2 数据预处理：PCA降维与归一化"""
    # PCA降维
    pca = PCA(n_components=N_COMPONENTS, svd_solver='randomized', whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((N_COMPONENTS, 19, 19))
    X_train_pca = pca.transform(X_train)  # (640, N_COMPONENTS)
    X_val_pca = pca.transform(X_val)  # (160, N_COMPONENTS)

    """Part3 SVM训练"""
    # SVC(C=1.8, class_weight='balanced', gamma=0.06) == 0.99  N_COMPONENTS = 25
    param_grid = {'C': [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5],
                  'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]}
    # param_grid = {'C': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11],
    #               'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
    clf = GridSearchCV(
        SVC(kernel=KERNEL, class_weight='balanced'), param_grid,
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    )
    clf = clf.fit(X_train_pca, y_train)
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    """可视化网格搜索结果"""
    results = clf.cv_results_
    # 创建一个 DataFrame 方便处理
    results_df = pd.DataFrame(results)
    # 提取需要的参数和评分
    param_C = results_df['param_C'].astype(float)
    param_gamma = results_df['param_gamma'].astype(float)
    mean_test_score = results_df['mean_test_score']
    # 创建一个透视表，用于热图
    heatmap_data = results_df.pivot_table(values='mean_test_score',
                                          index='param_gamma',
                                          columns='param_C')
    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Grid Search Mean Test Scores")
    plt.xlabel("C values")
    plt.ylabel("Gamma values")
    plt.show()

    """Part4 SVM预测与模型评估"""
    print("Predicting people's names on the test set...")
    y_pred = clf.predict(X_val_pca)
    print(classification_report(y_true=y_val, y_pred=y_pred, target_names=target_names))
    print(confusion_matrix(y_true=y_val, y_pred=y_pred))

    """Part 5 可视化"""
    #1 plot the result of the prediction on a portion of the test set
    prediction_titles = [title(y_pred, y_val, target_names, i) for i in range(y_pred.shape[0])]  # 对每个预测样本获取titles
    plot_gallery(X_val, prediction_titles, h, w)
    #2 plot the gallery of the most significative eigenfaces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w, n_row=1, n_col=5)
    plt.show()

    """Part 6 预测"""
    test_data_pca = pca.transform(test_data)
    test_data_pred = clf.predict(test_data_pca)
    # display_all_images(test_data, test_data_pred)

    with open('result.txt', 'w', encoding='utf-8') as f:
        for idx, result in enumerate(test_data_pred):
            f.write(str(idx + 1) + f' {result}\n')
