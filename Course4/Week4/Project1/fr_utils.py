#### PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py ####
# ------------------------------------------------------
#   fr_utils.py  —— TensorFlow 2.x 完整重写版本
#   适配吴恩达 DeepLearning.ai Facenet 项目
# ------------------------------------------------------
# fr_utils.py
import tensorflow as tf
import numpy as np
import os
import cv2
import h5py
from numpy import genfromtxt

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
from keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.models import Model


_FLOATX = 'float32'


# ------------------------------------------------------
# TF2.x variable（自动初始化，不需要 sess.run()）
# ------------------------------------------------------
def variable(value, dtype=_FLOATX, name=None):
    return tf.Variable(np.asarray(value, dtype=dtype), name=name)


# ------------------------------------------------------
# TF2.x shape
# ------------------------------------------------------
def shape(x):
    return x.shape


def square(x):
    return tf.square(x)


def zeros(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


# ------------------------------------------------------
# TF2.x concat
#   TF1: tf.concat(axis, tensors)
#   TF2: tf.concat(tensors, axis)
# ------------------------------------------------------
def concatenate(tensors, axis=-1):
    return tf.concat(tensors, axis=axis)


# ------------------------------------------------------
# LRN → 仍然可用
# ------------------------------------------------------
def LRN2D(x):
    return tf.nn.local_response_normalization(x, alpha=1e-4, beta=0.75)


# ------------------------------------------------------
# 卷积 + BN 组合层（与吴恩达课程一致）
# ------------------------------------------------------
def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):

    num = '' if cv2_out is None else '1'

    # conv1
    tensor = Conv2D(
        cv1_out, cv1_filter,
        strides=cv1_strides,
        data_format='channels_first',
        name=layer + '_conv' + num,
        use_bias=True
    )(x)

    tensor = BatchNormalization(
        axis=1,
        epsilon=0.00001,
        name=layer + '_bn' + num
    )(tensor)

    tensor = Activation('relu')(tensor)

    # 如果没有 padding
    if padding is None:
        return tensor

    # pad
    tensor = ZeroPadding2D(
        padding=padding,
        data_format='channels_first'
    )(tensor)

    # conv2
    if cv2_out is None:
        return tensor

    tensor = Conv2D(
        cv2_out, cv2_filter,
        strides=cv2_strides,
        data_format='channels_first',
        name=layer + '_conv2',
        use_bias=True
    )(tensor)

    tensor = BatchNormalization(
        axis=1,
        epsilon=0.00001,
        name=layer + '_bn2'
    )(tensor)

    tensor = Activation('relu')(tensor)

    return tensor


# ------------------------------------------------------
# 权重名称与 shape（原样保留）
# ------------------------------------------------------
WEIGHTS = [
  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
  'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  'inception_3a_pool_conv', 'inception_3a_pool_bn',
  'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  'inception_3b_pool_conv', 'inception_3b_pool_bn',
  'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  'inception_4a_pool_conv', 'inception_4a_pool_bn',
  'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  'inception_5a_pool_conv', 'inception_5a_pool_bn',
  'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  'inception_5b_pool_conv', 'inception_5b_pool_bn',
  'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
  'dense_layer'
]

conv_shape = {
  'conv1': [64, 3, 7, 7],
  'conv2': [64, 64, 1, 1],
  'conv3': [192, 64, 3, 3],
  'inception_3a_1x1_conv': [64, 192, 1, 1],
  'inception_3a_pool_conv': [32, 192, 1, 1],
  'inception_3a_5x5_conv1': [16, 192, 1, 1],
  'inception_3a_5x5_conv2': [32, 16, 5, 5],
  'inception_3a_3x3_conv1': [96, 192, 1, 1],
  'inception_3a_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_3x3_conv1': [96, 256, 1, 1],
  'inception_3b_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_5x5_conv1': [32, 256, 1, 1],
  'inception_3b_5x5_conv2': [64, 32, 5, 5],
  'inception_3b_pool_conv': [64, 256, 1, 1],
  'inception_3b_1x1_conv': [64, 256, 1, 1],
  'inception_3c_3x3_conv1': [128, 320, 1, 1],
  'inception_3c_3x3_conv2': [256, 128, 3, 3],
  'inception_3c_5x5_conv1': [32, 320, 1, 1],
  'inception_3c_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_3x3_conv1': [96, 640, 1, 1],
  'inception_4a_3x3_conv2': [192, 96, 3, 3],
  'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  'inception_4a_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_pool_conv': [128, 640, 1, 1],
  'inception_4a_1x1_conv': [256, 640, 1, 1],
  'inception_4e_3x3_conv1': [160, 640, 1, 1],
  'inception_4e_3x3_conv2': [256, 160, 3, 3],
  'inception_4e_5x5_conv1': [64, 640, 1, 1],
  'inception_4e_5x5_conv2': [128, 64, 5, 5],
  'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  'inception_5a_3x3_conv2': [384, 96, 3, 3],
  'inception_5a_pool_conv': [96, 1024, 1, 1],
  'inception_5a_1x1_conv': [256, 1024, 1, 1],
  'inception_5b_3x3_conv1': [96, 736, 1, 1],
  'inception_5b_3x3_conv2': [384, 96, 3, 3],
  'inception_5b_pool_conv': [96, 736, 1, 1],
  'inception_5b_1x1_conv': [256, 736, 1, 1],
}

# ------------------------------------------------------
# 加载权重
# ------------------------------------------------------
def load_weights_from_FaceNet(FRmodel):
    weights_dict = load_weights()

    for name in WEIGHTS:
        if FRmodel.get_layer(name) is not None:
            FRmodel.get_layer(name).set_weights(weights_dict[name])


def load_weights():
    dirPath = './weights'
    fileNames = [f for f in os.listdir(dirPath) if not f.startswith('.')]
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = os.path.join(dirPath, n)

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',')
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))

            conv_b = genfromtxt(paths[name + '_b'], delimiter=',')
            weights_dict[name] = [conv_w, conv_b]

        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',')
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',')
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',')
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',')

            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]

        elif 'dense' in name:
            dense_w = genfromtxt(os.path.join(dirPath, 'dense_w.csv'), delimiter=',')
            dense_w = np.reshape(dense_w, (128, 736)).T
            dense_b = genfromtxt(os.path.join(dirPath, 'dense_b.csv'), delimiter=',')
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict


# ------------------------------------------------------
# 数据集加载（Happy House）
# ------------------------------------------------------
def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    test_dataset  = h5py.File('datasets/test_happy.h5', "r")

    train_x = np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:]).reshape((1, -1))

    test_x  = np.array(test_dataset["test_set_x"][:])
    test_y  = np.array(test_dataset["test_set_y"][:]).reshape((1, -1))

    classes = np.array(test_dataset["list_classes"][:])

    return train_x, train_y, test_x, test_y, classes


# ------------------------------------------------------
# 转换图像为 FaceNet embedding
# ------------------------------------------------------
def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[..., ::-1]   # BGR → RGB

    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x = np.array([img])

    embedding = model.predict(x)
    return embedding

