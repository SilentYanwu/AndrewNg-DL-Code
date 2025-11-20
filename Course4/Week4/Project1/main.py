'''
Face Recognition System
参考何宽老师的CSDN博客：https://blog.csdn.net/u013733326/article/details/80767079 第一部分
原代码基于 TensorFlow 1.x，现已更新为 TensorFlow 2.x 版本
'''
# main.py
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, ZeroPadding2D, Activation, Input, 
                                     BatchNormalization, MaxPooling2D, AveragePooling2D,
                                     Lambda, Flatten, Dense)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K

import numpy as np
import time
import cv2
import os
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

import fr_utils
from inception_blocks_v2 import *

# TF2：默认 channels_last，不建议强行改 channels_first（很多预训练模型不支持）
# 如果必须使用 FaceNet 结构，可以保持不变
K.set_image_data_format('channels_first')

# 显示模型图
from tensorflow.keras.utils import plot_model

np.set_printoptions(threshold=np.inf)
tf.random.set_seed(1)

# ---------------------------------------------------------
#   Triplet Loss TF2.x 版本
# ---------------------------------------------------------
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    y_pred: shape = (batch, 3, 128)
    y_pred[:,0] = anchor
    y_pred[:,1] = positive
    y_pred[:,2] = negative
    """
    anchor, positive, negative = y_pred[:,0], y_pred[:,1], y_pred[:,2]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss


# ---------------------------------------------------------
#   构建 FaceNet 模型
# ---------------------------------------------------------
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("参数数量：", FRmodel.count_params())

# 可选：绘制模型
# plot_model(FRmodel, to_file='FaceRecogModel.png', show_shapes=True)


# ---------------------------------------------------------
#   编译 + 加载预训练权重（FaceNet）
# ---------------------------------------------------------
start_time = time.time()

FRmodel.compile(optimizer='adam', loss=triplet_loss)

fr_utils.load_weights_from_FaceNet(FRmodel)

end_time = time.time()
duration = end_time - start_time
print(f"执行了：{int(duration/60)}分{int(duration%60)}秒")


# ---------------------------------------------------------
#   建立数据库（encoding）
# ---------------------------------------------------------
database = {}
people = {
    "danielle":"images/danielle.png",
    "younes":"images/younes.jpg",
    "tian":"images/tian.jpg",
    "andrew":"images/andrew.jpg",
    "kian":"images/kian.jpg",
    "dan":"images/dan.jpg",
    "sebastiano":"images/sebastiano.jpg",
    "bertrand":"images/bertrand.jpg",
    "kevin":"images/kevin.jpg",
    "felix":"images/felix.jpg",
    "benoit":"images/benoit.jpg",
    "arnaud":"images/arnaud.jpg"
}

for name, path in people.items():
    database[name] = fr_utils.img_to_encoding(path, FRmodel)


# ---------------------------------------------------------
#   人脸验证 verify()
# ---------------------------------------------------------
def verify(image_path, identity, database, model):
    encoding = fr_utils.img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])

    if dist < 0.7:
        print(f"欢迎 {identity} 回家！ 距离：{dist}")
        return dist, True
    else:
        print(f"验证失败：与 {identity} 不符 (dist={dist})")
        return dist, False


verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg", "kian", database, FRmodel)


# ---------------------------------------------------------
#   人脸识别 who_is_it()
# ---------------------------------------------------------
def who_is_it(image_path, database, model):
    encoding = fr_utils.img_to_encoding(image_path, model)
    
    min_dist = 999
    identity = None

    for name, db_enc in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("抱歉，未找到匹配的身份。")
    else:
        print(f"姓名：{identity}  差距：{min_dist}")

    return min_dist, identity


who_is_it("images/camera_0.jpg", database, FRmodel)
