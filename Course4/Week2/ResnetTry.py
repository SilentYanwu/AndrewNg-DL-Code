"""
TF2.15 版本 ResNet50（AI实现）
参考：何宽老师 CSDN + Coursera 深度学习作业
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os,sys
# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)

# 在导入本地文件/模型之前调用
fix_paths()

import resnets_utils   

# ============================================================
# 1. Residual Blocks
# ============================================================

def identity_block(X, f, filters, stage, block):
    F1, F2, F3 = filters

    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base   = f"bn{stage}{block}_branch"

    X_shortcut = X

    # 1
    X = Conv2D(F1, (1,1), padding='valid',
               name=conv_name_base+"2a",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)

    # 2
    X = Conv2D(F2, (f,f), padding='same',
               name=conv_name_base+"2b",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    # 3
    X = Conv2D(F3, (1,1), padding='valid',
               name=conv_name_base+"2c",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)

    # Add shortcut
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    F1, F2, F3 = filters

    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base   = f"bn{stage}{block}_branch"

    X_shortcut = X

    # 1
    X = Conv2D(F1, (1,1), strides=(s,s), name=conv_name_base+"2a",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)

    # 2
    X = Conv2D(F2, (f,f), padding='same', name=conv_name_base+"2b",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    # 3
    X = Conv2D(F3, (1,1), padding='valid', name=conv_name_base+"2c",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)

    # Shortcut
    X_shortcut = Conv2D(F3, (1,1), strides=(s,s),
                        name=conv_name_base+"1",
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+"1")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


# ============================================================
# 2. ResNet50
# ============================================================

def ResNet50(input_shape=(64,64,3), classes=6):

    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)

    # stage1
    X = Conv2D(64, (7,7), strides=(2,2),
               name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    # stage2
    X = convolutional_block(X, 3, [64,64,256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64,64,256], stage=2, block='b')
    X = identity_block(X, 3, [64,64,256], stage=2, block='c')

    # stage3
    X = convolutional_block(X, 3, [128,128,512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # stage4
    X = convolutional_block(X, 3, [256,256,1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256,256,1024], stage=4, block='b')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='c')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='d')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='e')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='f')

    # stage5
    X = convolutional_block(X, 3, [512,512,2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512,512,2048], stage=5, block='b')
    X = identity_block(X, 3, [512,512,2048], stage=5, block='c')

    X = AveragePooling2D((2,2))(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax',
              name=f'fc{classes}',
              kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    return model


# ============================================================
# 3. 训练
# ============================================================

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

model = ResNet50(input_shape=(64,64,3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, Y_train, epochs=2, batch_size=32)

preds = model.evaluate(X_test, Y_test)
print("误差值 =", preds[0])
print("准确率 =", preds[1])
