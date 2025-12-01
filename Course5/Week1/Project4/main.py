# main.py
'''
爵士乐生成器 (TF 2.x 升级版)
代码参考：https://blog.csdn.net/u013733326/article/details/80890454#t3
'''
import numpy as np
import time
import sys
import IPython.display
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K # 保留 K 作为 Keras Backend 的简写

# --- 假定的自定义模块导入 ---
from music21 import * 
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
# -----------------------------

# --- 全局参数和初始化 ---
n_a = 64 # LSTM 激活单元的数量
Tx = 30 # 训练序列长度
Ty = 50 # 推理序列长度
n_values = 78 # 音乐数据中唯一值的数量 (音符/休止符/变化的总数)
m = 60 # 训练样本数量

# 加载数据 (假设 load_music_utils 已经升级到 TF2.x 兼容版本)
try:
    X, Y, n_values, indices_values = load_music_utils()
    # 打印数据信息
    print('✅ 数据加载成功。')
    print('输入 X 的形状:', X.shape)
    print('训练样本总数:', X.shape[0])
    print('Tx (序列长度):', X.shape[1])
    print('唯一值总数 (n_values):', n_values)
    print('输出 Y 的形状:', Y.shape)
except Exception as e:
    print(f"❌ 数据加载失败，请检查 music_utils 和 data_utils: {e}")
    # 设置默认值以使模型定义部分能够运行
    n_values = 78 


# 初始化器 (用于模型推理和训练)
x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

# --- 模型组件定义 (作为全局变量，便于在 djmodel 和 music_inference_model 中共享) ---
# 2.B: Reshape 层，将输入 (Batch, 78) 变为 (Batch, 1, 78)
reshapor = Reshape((1, n_values)) 
# 2.C: LSTM 单元，返回状态，便于在循环中传递 (return_sequences=False 是默认值)
LSTM_cell = LSTM(n_a, return_state = True, name='lstm_cell_shared')
# 2.D: Dense 层，输出 n_values 维度的概率分布 (使用 softmax)
densor = Dense(n_values, activation='softmax', name='densor_shared')


def djmodel(Tx, n_a, n_values, reshapor, LSTM_cell, densor):
    """
    实现训练模型 (Sequence-to-Sequence with Time-distributed Logic)
    
    参数：
        Tx -- 语料库的长度
        n_a -- 激活值的数量
        n_values -- 音乐数据中唯一值的数量
        reshapor, LSTM_cell, densor -- 共享 Keras 层对象
        
    返回：
        model -- Keras 模型实体
    """
    # 定义输入数据的维度
    X = Input((Tx, n_values), name='X_input')
    
    # 定义 a0, c0, 初始化隐藏状态
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
    a = a0
    c = c0
    
    # 第一步：创建一个空的outputs列表来保存 LSTM 的所有时间步的输出。
    outputs = []
    
    # 第二步：循环 Tx 次，处理每个时间步的输入
    for t in range(Tx):
        ## 2.A：使用 Lambda 层从 X 中选择第 't' 个时间步向量
        # Lambda 函数确保在 TensoFlow Graph 中正确切片
        x = Lambda(lambda x_full: x_full[:, t, :], output_shape=(n_values,))(X)
        
        ## 2.B：使用 reshapor 对 x 进行重构为 (Batch, 1, n_values)
        x = reshapor(x)
        
        ## 2.C：单步传播 (initial_state=[a, c] 传入上一步的状态)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        ## 2.D：使用 densor() 应用于 LSTM_Cell 的隐藏状态输出 'a'
        out = densor(a)
        
        ## 2.E：把预测值添加到 "outputs" 列表中
        outputs.append(out)
        
    # 第三步：创建模型实体
    model = Model(inputs=[X, a0, c0], outputs=outputs, name='DJ_Training_Model')
    
    return model

# 训练模型定义
model = djmodel(Tx=Tx, n_a=n_a, n_values=n_values, 
                reshapor=reshapor, LSTM_cell=LSTM_cell, densor=densor)

# 编译模型：使用 Adam 优化器与分类交叉熵损失。
# TF2.x 兼容性：Adam 和其他优化器不需要 K.tf.clip_by_value 等操作，但参数与 Keras 1/2 兼容。
opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 初始化 a0 和 c0，用于训练时的初始状态 (Batch size = m)
if 'X' in locals():
    m = X.shape[0] # 使用实际样本数量
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

# --- 模型训练 ---
print("--- 开始训练模型 ---")
start_time = time.time() # 使用 time.time() 代替 time.clock() (TF2.x/Python 3 推荐)

# 开始拟合
# Y 必须是列表形式，因为模型有 Tx 个输出
if 'X' in locals():
    # 确保 Y 是正确的列表格式 for Keras multi-output
    Y_list = list(Y) 
    
    model.fit([X, a0, c0], Y_list, epochs=100, batch_size=32)
    
    end_time = time.time()
    minium = end_time - start_time
    
    print("\n--- 训练结束 ---")
    print(f"执行了: {int(minium / 60)} 分 {int(minium % 60)} 秒")
# --------------------


def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50):
    """
    实现推理模型 (Generating Model)
    在推理模式下，模型使用上一个时间步的预测输出作为下一个时间步的输入。
    
    参数：
        LSTM_cell -- 训练过的 LSTM 单元，是 Keras 层对象。
        densor -- 训练过的 "densor"，是 Keras 层对象
        Ty -- 整数，生成的序列长度
        
    返回：
        inference_model -- Keras 模型实体
    """
    
    # 定义模型输入的维度
    x0 = Input(shape=(1, n_values), name='x0_input')
    
    # 定义 a0, c0，初始化隐藏状态
    a0 = Input(shape=(n_a,), name="a0_inf")
    c0 = Input(shape=(n_a,), name="c0_inf")
    a = a0
    c = c0
    x = x0
    
    # 步骤1：创建一个空的outputs列表来保存预测值。
    outputs = []
    
    # 步骤2：遍历 Ty，生成所有时间步的输出
    for t in range(Ty):
        
        # 步骤2.A：在 LSTM 中单步传播 (x 是上一步生成的 one-hot 向量)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # 步骤2.B：使用 densor() 应用于 LSTM_Cell 的隐藏状态输出 'a'
        out = densor(a)
        
        # 步骤2.C：预测值添加到 "outputs" 列表中
        outputs.append(out)
        
        # 步骤2.D：根据 'out' 选择下一个值，并将其 one-hot 编码设置为 'x'
        # one_hot 应该是一个自定义的 Lambda 函数，用于从概率分布中采样或选择 max
        # 这里使用 Lambda(one_hot) 确保它在 Keras 图中执行
        # one_hot 函数已在 data_utils 或 music_utils 中定义 (见前一个请求的实现)
        x = Lambda(one_hot)(out)
        
    # 创建模型实体
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs, name='Music_Inference_Model')
    
    return inference_model

# 获取推理模型实体 (硬编码 Ty = 50)
inference_model = music_inference_model(LSTM_cell, densor, n_values=n_values, n_a=n_a, Ty=Ty)

# 创建用于初始化 x 和 LSTM 状态变量 a 和 c 的零向量。
x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer, 
                       c_initializer=c_initializer):
    """
    使用推理模型进行预测和采样。
    
    参数：
        inference_model -- Keras 的实体模型
        ... 初始状态
    
    返回：
        results -- 生成值的独热编码向量，维度为(Ty, 78)
        indices -- 所生成值的索引矩阵，维度为(Ty, 1)
    """
    # 步骤1：模型来预测给定初始状态的输出序列 (返回 Ty 个概率分布矩阵)
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    
    # 步骤2：将“pred”转换为具有最大概率的索引数组 np.array()。
    # pred 是一个包含 Ty 个元素的列表，每个元素形状是 (1, 78)。
    # 我们将它们堆叠起来，并沿着最后一个轴 (axis=-1) 取最大值索引。
    pred_array = np.array(pred) # 形状: (Ty, 1, 78)
    indices = np.argmax(pred_array, axis=-1) # 形状: (Ty, 1)
    
    # 步骤3：将索引转换为它们的一个独热编码。
    results = to_categorical(indices.squeeze(), num_classes=n_values) # 形状: (Ty, 78)
    
    return results, indices

# --- 音乐生成和测试 ---
print("\n--- 开始音乐生成测试 ---")
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print(f"生成的序列长度 (Ty): {Ty}")
print(f"第 13 个值的索引 (np.argmax(results[12])): {np.argmax(results[12])}")
print(f"第 18 个值的索引 (np.argmax(results[17])): {np.argmax(results[17])}")
print(f"索引 12 到 17: {list(indices[12:18].flatten())}")

# 生成最终的 MIDI 文件 (假设 generate_music 已经升级到 TF2.x 兼容版本)
if 'inference_model' in locals():
    print("\n--- 开始生成 MIDI 文件 ---")
    out_stream = generate_music(inference_model, Ty=Ty)
    print("✅ MIDI 文件生成完毕: output/my_music.midi")


# --- 任务 3: 将生成的 MIDI 转化为 MP3 并播放 ---

# ⚠ 注意：MIDI 转 MP3 需要外部库和系统环境支持，例如 MuseScore 或 TiMidity++。
# 在标准的 Python/Colab/Jupyter 环境中，最简单的方法是使用 music21 库的 converter/midi.realtime 模块，
# 配合安装好的 MIDI 播放器/合成器。

def convert_midi_to_mp3_and_play(stream_obj):
    """
    将 music21 stream 对象转化为 MIDI 文件，尝试使用系统安装的工具将其渲染为 MP3 (或 wav) 并播放。
    
    参数:
        stream_obj -- music21.stream.Stream 对象
    """
    # 1. 保存 MIDI 文件 (已在 generate_music 中完成，但此处再次确认路径)
    midi_path = "output/my_music.midi"
    
    # 2. 尝试 MIDI 实时播放 (通常在本地安装了 music21 所依赖的播放器时有效)
    print("\n--- 尝试实时播放 MIDI ---")
    try:
        # music21.midi.realtime.StreamPlayer 依赖于底层 MIDI 合成器，如 fluidsynth 或 timidity
        sp = midi.realtime.StreamPlayer(stream_obj)
        sp.play()
        print("✅ MIDI 实时播放成功 (如果配置了 MIDI 合成器)。")
    except Exception as e:
        print(f"❌ MIDI 实时播放失败: {e}")
        
    # 3. 尝试渲染为 MP3/WAV (需要外部工具，如 MuseScore)
    # 这一步在纯 Python/TF 环境中非常复杂，通常需要安装和配置 MuseScore。
    # 以下代码仅为示意，实际运行需要正确的环境配置。
    print("\n--- 尝试将 MIDI 渲染为 MP3 ---")
    try:
        # 使用 music21 的 show() 方法尝试渲染，依赖于 MuseScore 或其他 MusicXML 渲染器
        # stream_obj.show('midi') # 尝试用默认 MIDI 播放器打开
        
        # 假设我们成功使用外部工具 (如 os.system("musescore /path/to/midi -o /path/to/mp3")) 
        # 转换为 output/my_music.mp3
        mp3_path = "output/my_music.mp3"
        print(f"💡 MIDI 转 MP3/WAV 渲染需要外部工具 (如 MuseScore 或 TiMidity++)。请手动运行或配置。")

        # 模拟播放 MP3 (如果文件存在)
        if os.path.exists(mp3_path):
             IPython.display.display(IPython.display.Audio(mp3_path))
             print("✅ MP3 文件播放成功。")
        else:
            # 仅播放 MIDI 文件作为替代
            IPython.display.display(IPython.display.Audio(midi_path))
            print("💡 无法找到 MP3 文件，尝试播放原始 MIDI 文件 (浏览器支持)。")
            
    except Exception as e:
        print(f"❌ MIDI 渲染或 MP3 播放失败: {e}")

# 播放生成的音乐
if 'out_stream' in locals():
    convert_midi_to_mp3_and_play(out_stream)