import numpy as np
import tensorflow as tf
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
import emo_utils

# 固定随机种子，保证结果可复现
np.random.seed(1)
tf.random.set_seed(1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Dropout, LSTM, Activation, Embedding
)
from tensorflow.keras.initializers import GlorotUniform

# 读取训练集和测试集
X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')

# 读取 GloVe 词向量
word_to_index, index_to_word, word_to_vec_map = \
    emo_utils.read_glove_vecs('data/glove.6B.50d.txt')

# 句子最大长度（按训练集最长句子）
maxLen = max(len(sentence.split()) for sentence in X_train)
def sentences_to_indices(X, word_to_index, max_len):
    """
    将句子数组转换为对应的单词索引矩阵（供 Embedding 层使用）

    参数：
        X -- 句子数组，shape: (m,)
        word_to_index -- 单词到索引的字典
        max_len -- 每个句子的最大长度

    返回：
        X_indices -- 索引矩阵，shape: (m, max_len)
    """

    m = X.shape[0]  # 样本数量
    X_indices = np.zeros((m, max_len), dtype=np.int32)

    for i in range(m):
        # 将句子转为小写并按空格分词
        sentence_words = X[i].lower().split()

        for j, word in enumerate(sentence_words):
            if j < max_len:
                X_indices[i, j] = word_to_index[word]

    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    创建并加载预训练 GloVe 向量的 Embedding 层（不可训练）

    参数：
        word_to_vec_map -- 单词到词向量的映射
        word_to_index -- 单词到索引的映射

    返回：
        embedding_layer -- 已加载权重的 Embedding 层
    """

    vocab_len = len(word_to_index) + 1      # 词表大小（+1 给 padding）
    emb_dim = next(iter(word_to_vec_map.values())).shape[0]

    # 初始化嵌入矩阵
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # 将 GloVe 向量写入嵌入矩阵
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # 创建 Embedding 层（冻结参数）
    embedding_layer = Embedding(
        input_dim=vocab_len,
        output_dim=emb_dim,
        trainable=False
    )

    # 手动 build 并加载权重
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Emojify-V2 模型（双层 LSTM）

    参数：
        input_shape -- 输入维度 (max_len,)
        word_to_vec_map -- GloVe 向量映射
        word_to_index -- 单词索引映射

    返回：
        model -- TF2.x Keras 模型
    """

    # 输入层：句子索引
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # 预训练词嵌入层
    embedding_layer = pretrained_embedding_layer(
        word_to_vec_map, word_to_index
    )

    # 嵌入输出：(batch, max_len, emb_dim)
    embeddings = embedding_layer(sentence_indices)

    # 第一层 LSTM（返回整个序列）
    X = LSTM(
        128,
        return_sequences=True,
        kernel_initializer=GlorotUniform(seed=1)
    )(embeddings)

    X = Dropout(0.5)(X)

    # 第二层 LSTM（只返回最后一个隐藏状态）
    X = LSTM(
        128,
        return_sequences=False,
        kernel_initializer=GlorotUniform(seed=1)
    )(X)

    X = Dropout(0.5)(X)

    # 全连接输出层（5 类情绪）
    X = Dense(5)(X)
    X = Activation('softmax')(X)

    # 构建模型
    model = Model(inputs=sentence_indices, outputs=X)

    return model


model = Emojify_V2(
    input_shape=(maxLen,),
    word_to_vec_map=word_to_vec_map,
    word_to_index=word_to_index
)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 训练集预处理
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = emo_utils.convert_to_one_hot(Y_train, C=5)

# 模型训练
model.fit(
    X_train_indices,
    Y_train_oh,
    epochs=50,
    batch_size=32,
    shuffle=True
)


# 测试集预处理
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
Y_test_oh = emo_utils.convert_to_one_hot(Y_test, C=5)

# 模型评估
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print("Test accuracy =", acc)

x_test = np.array(['you are so beautiful'])

X_test_indices = sentences_to_indices(
    x_test, word_to_index, maxLen
)

pred = model.predict(X_test_indices)
emoji_index = np.argmax(pred)

print(
    x_test[0],
    emo_utils.label_to_emoji(emoji_index)
)
