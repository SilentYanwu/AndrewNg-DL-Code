'''
莎士比亚诗歌生成器
使用预训练的LSTM模型生成莎士比亚风格的诗歌。
代码参考：https://blog.csdn.net/u013733326/article/details/80890454#t3
'''
import os
import warnings
import logging

# 彻底抑制所有警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示 ERROR 级别日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 提示
warnings.filterwarnings('ignore')  # 忽略Python警告
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # 设置TensorFlow日志级别

# 导入TensorFlow并设置日志级别
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# 加载包
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import sys,os,io
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

def build_data(text, Tx=40, stride=3):
    """
    通过在文本语料库上扫描大小为Tx的窗口来创建训练集，步长为3。
    
    参数:
    text -- 字符串，莎士比亚诗歌的语料库
    Tx -- 序列长度，一个训练示例中的时间步数（或字符数）
    stride -- 扫描时窗口移动的步长
    
    返回:
    X -- 训练示例列表
    Y -- 训练标签列表
    """
    
    X = []
    Y = []

    ### 开始代码 ### (约3行)
    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])  # 获取输入序列
        Y.append(text[i + Tx])     # 获取下一个字符作为标签
    ### 结束代码 ###
    
    print('训练示例数量:', len(X))
    
    return X, Y


def vectorization(X, Y, n_x, char_indices, Tx=40):
    """
    将X和Y（列表）转换为可以提供给循环神经网络的数组。
    
    参数:
    X -- 输入序列列表
    Y -- 标签列表
    Tx -- 整数，序列长度
    
    返回:
    x -- 形状为(m, Tx, len(chars))的数组
    y -- 形状为(m, len(chars))的数组
    """
    
    m = len(X)  # 训练示例数量
    x = np.zeros((m, Tx, n_x), dtype=np.bool_)  # 初始化输入数组
    y = np.zeros((m, n_x), dtype=np.bool_)      # 初始化输出数组
    
    # 将字符转换为one-hot编码
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1  # 设置对应字符位置为1
        y[i, char_indices[Y[i]]] = 1         # 设置标签字符位置为1
        
    return x, y 


def sample(preds, temperature=1.0):
    """
    从概率数组中采样索引的辅助函数
    
    参数:
    preds -- 预测概率数组
    temperature -- 温度参数，控制随机性程度
    
    返回:
    采样的字符索引
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature      # 应用温度参数
    exp_preds = np.exp(preds)                # 指数运算
    preds = exp_preds / np.sum(exp_preds)    # 重新归一化
    probas = np.random.multinomial(1, preds, 1)  # 多项式采样
    out = np.random.choice(range(len(preds)), p=probas.ravel())  # 根据概率选择字符
    return out


def on_epoch_end(epoch, logs):
    """在每个epoch结束时调用的函数。打印生成的文本。"""
    pass


def main():
    """主函数"""
    # 在导入本地文件/模型之前调用
    fix_paths()
    print("加载文本数据...")
    try:
        # 读取莎士比亚文本数据并转换为小写
        text = io.open('shakespeare.txt', encoding='utf-8').read().lower()
        print(f"文本长度: {len(text)} 字符")
    except FileNotFoundError:
        print("错误：找不到 'shakespeare.txt' 文件")
        print("请确保文件在当前目录下")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 文本处理参数
    Tx = 40  # 序列长度
    chars = sorted(list(set(text)))  # 获取所有唯一字符并排序
    char_indices = dict((c, i) for i, c in enumerate(chars))  # 字符到索引的映射
    indices_char = dict((i, c) for i, c in enumerate(chars))  # 索引到字符的映射
    
    print(f"唯一字符数量: {len(chars)}")

    print("创建训练集...")
    X, Y = build_data(text, Tx, stride=3)  # 构建训练数据
    print("向量化训练集...")
    x, y = vectorization(X, Y, n_x=len(chars), char_indices=char_indices)  # 转换为one-hot编码
    
    print("加载模型...")
    try:
        model = load_model('models/model_shakespeare_kiank_350_epoch.h5', compile=False)  # 加载预训练模型
        print("模型加载成功！")
    except FileNotFoundError:
        print("错误：找不到模型文件 'models/model_shakespeare_kiank_350_epoch.h5'")
        print("请确保模型文件在 models 目录下")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 调用生成函数
    generate_output(model, chars, char_indices, indices_char, Tx)


def generate_output(model, chars, char_indices, indices_char, Tx):
    """
    生成诗歌的主函数
    接收用户输入，然后使用模型完成诗歌创作
    
    参数:
    model -- 训练好的模型
    chars -- 字符列表
    char_indices -- 字符到索引的映射
    indices_char -- 索引到字符的映射
    Tx -- 序列长度
    """
    generated = ''  # 存储生成的文本
    # 获取用户输入的诗歌开头
    usr_input = input("请输入你的诗歌开头，莎士比亚机器将完成它。你的输入是: ")
    
    # 用零填充句子到Tx个字符
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input 

    # 输出提示信息
    print("\n\n这是你的诗歌: \n")
    print(usr_input, end='', flush=True)
    
    # 生成400个字符
    for i in range(400):
        # 准备预测输入
        x_pred = np.zeros((1, Tx, len(chars)))

        # 将当前句子转换为one-hot编码
        for t, char in enumerate(sentence):
            if char != '0':  # 忽略填充的零
                x_pred[0, t, char_indices[char]] = 1.

        # 进行预测
        preds = model.predict(x_pred, verbose=0)[0]
        # 采样下一个字符（传递字符数量参数）
        next_index = sample(preds, temperature=1.0)
        next_char = indices_char[next_index]

        # 更新生成的文本和当前句子
        generated += next_char
        sentence = sentence[1:] + next_char  # 滑动窗口

        # 输出下一个字符
        print(next_char, end='', flush=True)

        # 如果遇到换行符，继续生成
        if next_char == '\n':
            continue
    
    print("\n\n诗歌生成完成！")


# 主程序入口
if __name__ == "__main__":
    main()