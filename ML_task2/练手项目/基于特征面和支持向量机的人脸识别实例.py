"""
    demo1——学习使用sklearn中的svm完成人脸分类任务
    https://scikit-learn.org.cn/view/187.html
    https://blog.csdn.net/woshicver/article/details/112001192
"""
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


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
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
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
        predicted: Bush
        true:      Bush
    """
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


"""Part1 数据加载 生成训练集 和 测试集"""
"""
# 数据集信息
# 1 查看dataset的属性 
print(lfw_people.keys())  # ['data', 'images', 'target', 'target_names', 'DESCR']
# 2 data属性包含扁平化的图像，作为一个2D NumPy数组
print(lfw_people.data.shape)  # (1288, 1850)
# 3 images属性包含原始图像，作为一个3D NumPy数组
print(lfw_people.images.shape)  # (1288, 50, 37)
# 4 target属性标识了每一个样本属于哪个类别
print(lfw_people.target.shape)  # (1288,) 样本数目
# 5 target_names属性表示了每个类别的名称
print(lfw_people.target_names)  # (7,) 有7类样本
"""
# step1 将数据加载到numpy数组中
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  # 加上resize可以让图片尺寸从(62, 47)->(50, 37)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data  # 样本集 (1288, 1850) ***
y = lfw_people.target  # 正确类别 (1288,) ***
target_names = lfw_people.target_names  # 类别名称(这里是不同人名)
n_features = X.shape[1]  # 特征维度1850
n_classes = target_names.shape[0]  # 类别个数

# # step2 显式查看图片数据集
# images = lfw_people.images  # 获取图片数据
# target = lfw_people.target  # 获取对应的标签
# target_names = lfw_people.target_names  # 获取对应的标签名称
# # 选择一个图片进行显示，例如显示第0张图片
# image = images[0]
# label = target_names[target[0]]
# # 创建一个图形和子图
# fig, ax = plt.subplots()
# # 显示图片
# ax.imshow(image, cmap=plt.cm.gray)  # 使用灰度颜色映射
# ax.set_title(f'Label: {label}')
# ax.axis('off')  # 不显示坐标轴
# # 显示图形
# plt.show()

# step3 划分训练集 和 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)  # (966, 1850) (322, 1850) (966,) (322,)

"""Part2 数据预处理"""
# PCA降维
n_components = 150  # 降维后的维数
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(
    X_train)  # ①pca实例构建；②使用训练数据对PCA模型进行拟合。PCA的拟合过程包括计算协方差矩阵、特征值和特征向量。
eigenfaces = pca.components_.reshape((n_components, h, w))
"""
代码功能：这行代码的目的是将PCA的主成分重塑为2D图像的形式，以便于后续的可视化。
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    plt.show()
代码解释：
    ① pca.components_返回的是一个数组，其中包含了PCA算法在降维过程中找到的主成分（特征向量）。在人脸识别任务中，PCA的主成分被称为“特征脸”。
    ② 通过将PCA的主成分重塑为2D图像的形式，可以得到一组“特征脸”图像。通过可视化这些“特征脸”图像，可以直观地理解PCA在人脸识别任务中所起的作用。
    ③ 需要注意的是，在人脸识别任务中，“特征脸”并不是原始人脸图像的直接表示。“特征脸”是通过PCA算法从原始人脸图像中提取出来的低维表示。在进行人脸识别时，通常会使用“特征脸”来进行匹配和识别。
"""

# 经过降维后的X_train 和 X_test，使用这个来训练
# 可以看到，我们在做PCA降维和归一化这种数据的预处理，不仅需要对训练样本使用，还要对预测样本使用，不能只对一个！
X_train_pca = pca.transform(X_train)  # (966, 150)
X_test_pca = pca.transform(X_test)  # (322, 150)

"""Part3 SVM Training"""
# SVM中的两个超参数：C是对错分样本的惩罚，C越大越不希望分错，margin越小；gamma是rbf核函数的超参数
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# 定义 网格搜索GridSearchCV实例clf，使用rbf核函数；并且每个类别的样本数量差不多，所以用balance
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
# 将GridSearchCV对象拟合到训练数据（X_train_pca, y_train）。此过程将穷举搜索参数网格中的所有超参数组合，以找到最佳的超参数组合。
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

"""Part4 SVM预测与模型评估"""
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

"""Part 5 可视化"""
# plot the result of the prediction on a portion of the test set
prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
print(prediction_titles)
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
