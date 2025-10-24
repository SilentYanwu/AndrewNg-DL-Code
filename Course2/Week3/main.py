'''
参考教程：https://blog.csdn.net/u013733326/article/details/79971488
但是由于没有 TensorFlow 2.x 版本的代码，故自行改写
本次作业三 - TensorFlow 2.x 版本三层神经网络
保持接近原始代码风格
这份代码跑起来时间过长，而且本人也没有Tensorflow的GPU环境，所以没有对代码进行优化
对于像图片处理（引入cv2等库）等其他功能放在Try2中实现，Try2由torch实现
仅供参考学习
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import time
import matplotlib.image as mpimg # mpimg 用于读取图片

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

fix_paths()

import tf_utils

# 设置随机种子
np.random.seed(1)
tf.random.set_seed(1)

# 加载数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

# 显示样本
index = 11
plt.imshow(X_train_orig[index])
print("Y = " + str(np.squeeze(Y_train_orig[:, index])))
plt.show()

# 数据预处理
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# 归一化数据
X_train = X_train_flatten / 255.0
X_test = X_test_flatten / 255.0

# 转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)

print("训练集样本数 = " + str(X_train.shape[1]))
print("测试集样本数 = " + str(X_test.shape[1]))
print("X_train.shape: " + str(X_train.shape))
print("Y_train.shape: " + str(Y_train.shape))
print("X_test.shape: " + str(X_test.shape))
print("Y_test.shape: " + str(Y_test.shape))


def initialize_parameters():
    """
    初始化神经网络的参数
    """
    initializer = tf.keras.initializers.GlorotUniform(seed=1)
    
    W1 = tf.Variable(initializer(shape=(25, 12288)), name="W1")
    b1 = tf.Variable(tf.zeros(shape=(25, 1)), name="b1")
    W2 = tf.Variable(initializer(shape=(12, 25)), name="W2")
    b2 = tf.Variable(tf.zeros(shape=(12, 1)), name="b2")
    W3 = tf.Variable(initializer(shape=(6, 12)), name="W3")
    b3 = tf.Variable(tf.zeros(shape=(6, 1)), name="b3")
    
    parameters = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2, 
        "W3": W3, "b3": b3
    }
    
    return parameters


def forward_propagation(X, parameters):
    """
    前向传播
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3


def compute_cost(Z3, Y):
    """
    计算成本函数
    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    return cost


def model(X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500, minibatch_size=32,
          print_cost=True, is_plot=True):
    """
    三层神经网络模型 - TensorFlow 2.x 版本
    """
    # 获取数据维度
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    # 初始化参数
    parameters = initialize_parameters()
    
    # 创建优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_cost = 0.0
        num_minibatches = int(m / minibatch_size)
        seed = epoch + 1
        minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)
        
        for minibatch in minibatches:
            # 选择一个小批量
            minibatch_X, minibatch_Y = minibatch
            
            # 使用GradientTape跟踪操作
            with tf.GradientTape() as tape:
                # 前向传播
                Z3 = forward_propagation(minibatch_X, parameters)
                # 计算成本
                cost = compute_cost(Z3, minibatch_Y)
            
            # 计算梯度
            grads = tape.gradient(cost, list(parameters.values()))
            
            # 更新参数
            optimizer.apply_gradients(zip(grads, list(parameters.values())))
            
            # 累加成本
            epoch_cost += cost.numpy() / num_minibatches
        
        # 记录成本
        if epoch % 5 == 0:
            costs.append(epoch_cost)
        
        # 打印成本
        if print_cost and epoch % 100 == 0:
            print(f"epoch = {epoch}    epoch_cost = {epoch_cost:.6f}")
    
    # 绘制成本曲线
    if is_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title(f"Learning rate = {learning_rate}")
        plt.show()
    
    # 计算准确率
    def calculate_accuracy(X, Y, parameters):
        Z3 = forward_propagation(X, parameters)
        predictions = tf.argmax(Z3, axis=0)
        labels = tf.argmax(Y, axis=0)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        return accuracy.numpy()
    
    train_accuracy = calculate_accuracy(X_train, Y_train, parameters)
    test_accuracy = calculate_accuracy(X_test, Y_test, parameters)
    
    print(f"训练集的准确率: {train_accuracy:.4f}")
    print(f"测试集的准确率: {test_accuracy:.4f}")
    
    return parameters


# 开始训练
print("开始训练三层神经网络...")
start_time = time.time()

# 训练模型
parameters = model(X_train, Y_train, X_test, Y_test, 
                  learning_rate=0.0001, 
                  num_epochs=1500, 
                  minibatch_size=32)

end_time = time.time()
print(f"执行时间 = {end_time - start_time:.2f} 秒")

# 测试训练好的模型
def predict(X, parameters):
    """
    使用训练好的模型进行预测
    """
    Z3 = forward_propagation(X, parameters)
    predictions = tf.argmax(Z3, axis=0)
    return predictions.numpy()

# 在测试集上进行预测
test_predictions = predict(X_test, parameters)
test_labels = tf.argmax(Y_test, axis=0).numpy()

# 计算测试准确率
test_accuracy = np.mean(test_predictions == test_labels)
print(f"最终测试集准确率: {test_accuracy:.4f}")

# 显示一些预测结果
print("\n前10个测试样本的预测结果:")
print("预测:", test_predictions[:10])
print("真实:", test_labels[:10])



#这是网上找到的图片
my_image1 = "5.png"                                            #定义图片名称
fileName1 = "images/" + my_image1                              #图片地址
image1 = mpimg.imread(fileName1)                               #读取图片
plt.imshow(image1)                                             #显示图片
my_image1 = image1.reshape(1,64 * 64 * 3).T                    #重构图片
my_image_prediction = tf_utils.predict(my_image1, parameters)  #开始预测
print("预测结果: y = " + str(np.squeeze(my_image_prediction)))
