import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# 1. 生成线性可分样本集
np.random.seed(0)
X_linear = np.random.randn(60, 2)
y_linear = np.ones(60)
X_linear[30:] = np.random.randn(30, 2) + np.array([3, 3])
y_linear[30:] = -1

# 2. 生成线性不可分样本集
X_nonlinear, y_nonlinear = datasets.make_circles(n_samples=60, factor=0.5, noise=0.1)

# 3. 使用SVM进行分类
def plot_svm(X, y, title):
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X, y)

    # 画出决策边界和支持向量
    plt.figure()
    ax = plt.gca()

    # 画出支持向量
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    # 画出分类平面
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 画出支持面
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
    ax.contourf(xx, yy, Z, levels=[-1, 0], colors='lightgray', alpha=0.5)
    ax.contourf(xx, yy, Z, levels=[0, 1], colors='gray', alpha=0.5)

    # 画出样本点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# 4. 可视化线性可分样本集
plot_svm(X_linear, y_linear, 'Linear Separability')

# 5. 可视化线性不可分样本集
clf_nonlinear = svm.SVC(kernel='rbf', C=1)
clf_nonlinear.fit(X_nonlinear, y_nonlinear)

# 画出支持向量
plt.figure()
ax = plt.gca()
plt.scatter(clf_nonlinear.support_vectors_[:, 0], clf_nonlinear.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

# 画出分类平面
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf_nonlinear.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 画出支持面
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
ax.contourf(xx, yy, Z, levels=[-1, 0], colors='lightgray', alpha=0.5)
ax.contourf(xx, yy, Z, levels=[0, 1], colors='gray', alpha=0.5)

# 画出样本点
plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear, cmap=plt.cm.Paired)
plt.title('Non-linear Separability')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
