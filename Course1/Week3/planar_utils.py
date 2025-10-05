import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(model, X, y):
    """绘制模型决策边界"""
    # 设置坐标轴范围并添加边距
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01  # 网格步长
    
    # 生成网格点坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                         np.arange(y_min, y_max, h))
    
    # 对网格中的所有点进行预测
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

def sigmoid(x):
    """Sigmoid激活函数"""
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset():
    """加载花瓣形状的二维数据集"""
    np.random.seed(1)  # 设置随机种子保证可重复性
    m = 400  # 样本数量
    N = int(m/2)  # 每类样本数
    D = 2  # 特征维度
    X = np.zeros((m, D))  # 特征矩阵
    Y = np.zeros((m, 1), dtype='uint8')  # 标签向量 (0:红色, 1:蓝色)
    a = 4  # 花瓣最大半径

    # 为两个类别生成数据
    for j in range(2):
        ix = range(N*j, N*(j+1))  # 当前类别的索引范围
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N)*0.2  # 角度θ
        r = a*np.sin(4*t) + np.random.randn(N)*0.2  # 半径r
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]  # 极坐标转笛卡尔坐标
        Y[ix] = j  # 设置类别标签

    # 转置为(特征数, 样本数)格式
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    """加载额外的测试数据集"""
    N = 200  # 每个数据集的样本数
    
    # 生成各种形状的数据集
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)  # 无结构随机数据

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure