'''
Course5.Week2.Project1.main 的 Docstring
代码参考：https://blog.csdn.net/u013733326/article/details/83341643
'''
import numpy as np

def read_glove_vecs(glove_file):
    """
    读取 GloVe 向量文件

    Arguments:
        glove_file -- 如 'glove.6B.50d.txt'

    Returns:
        words -- set，词汇表
        word_to_vec_map -- dict，word -> vector
    """
    words = set()
    word_to_vec_map = {}

    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            vec = np.array(line[1:], dtype=np.float64)

            words.add(word)
            word_to_vec_map[word] = vec

    return words, word_to_vec_map


def cosine_similarity(u, v):
    """
    计算余弦相似度
    """
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    return dot / (norm_u * norm_v)


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    A : B :: C : ?
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a = word_to_vec_map[word_a]
    e_b = word_to_vec_map[word_b]
    e_c = word_to_vec_map[word_c]

    max_sim = -1e9
    best_word = None

    for word in word_to_vec_map.keys():
        if word in [word_a, word_b, word_c]:
            continue

        cosine_sim = cosine_similarity(
            e_b - e_a,
            word_to_vec_map[word] - e_c
        )

        if cosine_sim > max_sim:
            max_sim = cosine_sim
            best_word = word

    return best_word

words, word_to_vec_map = read_glove_vecs(
    'data/glove.6B.50d.txt'
)

print(word_to_vec_map['hello'])

word1 = 'man'
word2 = 'computers'
word3 = 'woman'

print("使用词向量完成类比：")
print("{} -> {} :: {} -> {}".format(
    word1, word2, word3,
    complete_analogy(
        word1, word2, word3, word_to_vec_map
    )
))
print("===========================      ===========================")

# 计算 gender 向量（man-woman）
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)


# 计算每个词与g的余弦相似度
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))

word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w,  cosine_similarity(word_to_vec_map[w], g))


def neutralize(word, g, word_to_vec_map):
    """
    通过将“word”投影到与偏置轴正交的空间上，消除了“word”的偏差。
    该函数确保“word”在性别的子空间中的值为0
    
    参数：
        word -- 待消除偏差的字符串
        g -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射
        
    返回：
        e_debiased -- 消除了偏差的向量。
    """
    
    # 根据word选择对应的词向量
    e = word_to_vec_map[word]
    
    # 根据公式2计算e_biascomponent
    e_biascomponent = np.divide(np.dot(e, g), np.square(np.linalg.norm(g))) * g
    
    # 根据公式3计算e_debiased
    e_debiased = e - e_biascomponent
    
    return e_debiased

e = "receptionist"
print("去偏差前{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(word_to_vec_map["receptionist"], g)))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("去偏差后{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(e_debiased, g)))
if abs(cosine_similarity(e_debiased, g)) < 1e-15:
    print("成功去偏差")

# 计算每个词与g的余弦相似度
def equalize(pair, bias_axis, word_to_vec_map):
    """
    通过遵循上图中所描述的均衡方法来消除性别偏差。
    
    参数：
        pair -- 要消除性别偏差的词组，比如 ("actress", "actor") 
        bias_axis -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射
    
    返回：
        e_1 -- 第一个词的词向量
        e_2 -- 第二个词的词向量
    """
    # 第1步：获取词向量
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # 第2步：计算w1与w2的均值
    mu = (e_w1 + e_w2) / 2.0
    
    # 第3步：计算mu在偏置轴与正交轴上的投影
    mu_B = np.divide(np.dot(mu, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    mu_orth = mu - mu_B
    
    # 第4步：使用公式7、8计算e_w1B 与 e_w2B
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    
    # 第5步：根据公式9、10调整e_w1B 与 e_w2B的偏置部分
    corrected_e_w1B = np.sqrt(np.abs(1-np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w1B-mu_B, np.abs(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1-np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w2B-mu_B, np.abs(e_w2 - mu_orth - mu_B))
    
    # 第6步： 使e1和e2等于它们修正后的投影之和，从而消除偏差
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
    
    return e1, e2

print("==========均衡校正前==========")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("\n==========均衡校正后==========")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
