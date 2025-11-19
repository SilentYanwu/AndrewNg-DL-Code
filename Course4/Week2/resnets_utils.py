'''
通过AI转化后的ResNet工具函数,适配TF2.x和Keras
我仅仅修改了相关代码，并没有直接运行测试。
'''
import os
import numpy as np
import tensorflow as tf
import h5py
import math


# -------------------------
# 1. 数据集加载函数
# -------------------------
def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



# -------------------------
# 2. Mini-batch 构造函数
# -------------------------
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []

    # Step 1: Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition
    num_complete_minibatches = m // mini_batch_size

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # 最后不足一个 batch 的部分
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches



# -------------------------
# 3. One-hot 转换
# -------------------------
def convert_to_one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)].T



# -------------------------
# 4. 前向传播（预测用）
# -------------------------
def forward_propagation_for_predict(X, parameters):
    """
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    TF2 版本：使用 Eager mode，直接返回结果
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)

    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)

    Z3 = tf.matmul(W3, A2) + b3

    return Z3



# -------------------------
# 5. 预测函数（TF2 纯 Eager）
# -------------------------
def predict(X, parameters):
    """
    输入:
        X shape = (12288, 1)
    返回:
        预测类别 index
    """

    # TF2 中无需 placeholder，只需确保为 tensor
    X = tf.constant(X, dtype=tf.float32)

    # 将参数转为 TF2 tensor
    params = {
        key: tf.constant(value, dtype=tf.float32)
        for key, value in parameters.items()
    }

    # 前向传播
    Z3 = forward_propagation_for_predict(X, params)

    # Softmax + argmax
    prediction = tf.argmax(Z3, axis=0).numpy()

    return prediction
