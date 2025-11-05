# cnn_utils.py
import math
import numpy as np
import h5py
import tensorflow as tf

def load_dataset():
    """加载手势识别数据集"""
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 标签

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])  # 类别名称

    # 调整形状
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """将(X, Y)划分为小批量数据块"""
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # 处理最后一批不足的情况
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


def convert_to_one_hot(Y, C):
    """独热编码"""
    return np.eye(C)[Y.reshape(-1)].T


def predict(model, X):
    """
    使用训练好的Keras模型进行预测
    Arguments:
        model -- 训练好的tf.keras模型
        X -- 输入图像数据 shape=(m,64,64,3)
    Returns:
        preds -- 预测的类别索引数组
    """
    logits = model(X, training=False)
    preds = tf.argmax(logits, axis=1).numpy()
    return preds
