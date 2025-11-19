"""
本项目代码源于何宽老师的：
https://blog.csdn.net/u013733326/article/details/80250818 第一部分 Kears入门

原作业是基于 TensorFlow 1.x 的 Keras 接口编写，
此版本改写为 TensorFlow 2.x，并使用 tf.keras
"""

import numpy as np
import tensorflow as tf
# 这里可以运行 大概是因为TF和kears的内部问题，正常可以运行。
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv2D, ZeroPadding2D, BatchNormalization,
    Activation, MaxPooling2D, Flatten, Dense
)
from tensorflow.keras.preprocessing import image
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

import kt_utils      

# ------------------------------------------------------
# 1. 加载数据
# ------------------------------------------------------
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

# Normalize
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# 转置标签 shape (1,m) → (m,1)
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples =", X_train.shape[0])
print("number of test examples =", X_test.shape[0])
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# ------------------------------------------------------
# 2. 模型定义（TF2.x 版本）
# ------------------------------------------------------
def HappyModel(input_shape):
    """
    检测笑容的简单 CNN 模型（TF2.x）
    """
    X_input = Input(input_shape)

    # Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> ReLU
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # Pool
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # Flatten + FC
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name="HappyModel")
    return model


# ------------------------------------------------------
# 3. 模型实例
# ------------------------------------------------------
happy_model = HappyModel((64, 64, 3))
happy_model.summary()

# ------------------------------------------------------
# 4. 编译模型
# ------------------------------------------------------
happy_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# ------------------------------------------------------
# 5. 训练
# ------------------------------------------------------
happy_model.fit(X_train, Y_train,
                epochs=20,
                batch_size=32)

# ------------------------------------------------------
# 6. 评估
# ------------------------------------------------------
loss, acc = happy_model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc:.4f}")

# ------------------------------------------------------
# 7. 单张图片预测
# ------------------------------------------------------
img_path = 'images/test.jpg'   # 修改为你的路径
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0               # TF2 推荐直接归一化

prediction = happy_model.predict(x)
print("Smile probability:", prediction)
