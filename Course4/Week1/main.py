'''
tensorflow实现三层卷积神经网络，
来实现Course2第三周由于DNN比较简单而无法很好处理图像数据的问题：0-5的手势识别
本内容CSDN何宽大大博客：https://blog.csdn.net/u013733326/article/details/80086090 第二部分
本项目是tensorflow 1.x版本编写，如需在tensorflow 2.x版本运行，请使用tf.compat.v1进行兼容性处理，不过这个我不会啦QaQ
因此我决定将代码迁移到tensorflow 2.x版本，顺便给吴老师给出cnn_utils.py文件也迁移一下
不过本人不打算学习tensorflow，因此代码跑通即可
仅供参考
'''
"""
tensorflow实现三层卷积神经网络（迁移到TF 2.x版）
任务：0-5手势识别
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models # 不要在意这里的报错，反正tf2.15的环境可以用
import cnn_utils
import os, sys
from PIL import Image
import matplotlib.pyplot as plt

# 设置 Matplotlib 使用支持中文的字体（Windows 推荐 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# =============================
# 路径修复
# =============================
def fix_paths():
    """修复导入路径和文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

fix_paths()


# =============================
# 数据加载与预处理
# =============================
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()

# 去掉默认展示图片
# index = 6
# plt.imshow(X_train_orig[index])
# print("y =", np.squeeze(Y_train_orig[:, index]))

# 归一化
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# One-hot 编码
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples =", X_train.shape[0])
print("number of test examples =", X_test.shape[0])
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# =============================
# 模型定义
# =============================
def build_model(input_shape=(64, 64, 3), classes=6):
    model = models.Sequential([
        layers.Conv2D(8, (4, 4), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same'),
        layers.Conv2D(16, (2, 2), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'),
        layers.Flatten(),
        layers.Dense(classes, activation=None)  # logits
    ])
    return model


model = build_model()

# =============================
# 模型编译与训练
# =============================
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.009)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

history = model.fit(
    X_train, Y_train,
    epochs=150,
    batch_size=64,
    validation_data=(X_test, Y_test),
    verbose=1
)


# =============================
# 成本曲线绘制
# =============================
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend()
plt.title('Learning rate = 0.009')
plt.show()


# =============================
# 模型评估
# =============================
train_acc = model.evaluate(X_train, Y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, Y_test, verbose=0)[1]

print(f"✅ 训练集准确率: {train_acc:.4f}")
print(f"✅ 测试集准确率: {test_acc:.4f}")


# =============================
# 用户选择图片进行预测
# =============================
def predict_custom_image(model, img_path):
    """
    用户选择图片路径 -> 预处理 -> 模型预测
    """
    try:
        img = Image.open(img_path).resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1,64,64,3)
        logits = model.predict(img_array)
        pred = np.argmax(logits, axis=1)[0]
        plt.imshow(img)
        plt.title(f"预测结果: {pred}")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"❌ 无法读取图片: {e}")


# 让用户输入图片路径
user_img = input("👉 请输入要预测的图片路径（如: test_image.jpg）：").strip()
if os.path.exists(user_img):
    predict_custom_image(model, user_img)
else:
    print("⚠️ 未提供有效图片路径，跳过预测。")
