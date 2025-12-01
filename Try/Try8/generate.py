'''
generate.py
莎士比亚诗歌生成器 - 生成器
使用PyTorch LSTM模型生成莎士比亚风格的诗歌。
'''
import os,sys
import warnings
import io
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 彻底抑制所有警告和日志
warnings.filterwarnings('ignore')

# --- 1. 设备和配置 ---
# 检查是否有可用的 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# --- 2. PyTorch 模型定义 ---（与trainmodel.py中的定义相同）
class ShakespeareModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        """
        初始化LSTM模型。
        
        参数:
        input_size -- 词汇表大小 (len(chars))
        hidden_size -- LSTM隐藏层大小 (e.g., 128)
        num_layers -- LSTM层数 (e.g., 2)
        output_size -- 词汇表大小 (len(chars))
        dropout_rate -- Dropout比例
        """
        super(ShakespeareModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层：batch_first=True 表示输入张量的形状是 (batch, sequence, features)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 全连接层 (输出层)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        前向传播。
        
        参数:
        x -- 输入张量，形状为 (batch_size, seq_len, input_size)
        hidden -- 初始隐藏状态和细胞状态
        
        返回:
        out -- 输出张量，形状为 (batch_size, seq_len, output_size)
        hidden -- 最终隐藏状态和细胞状态
        """
        # x 形状: (batch_size, Tx, n_x)
        out, hidden = self.lstm(x, hidden)
        
        # out 形状: (batch_size, Tx, hidden_size)
        # 我们只关心最后一个时间步的输出，但字符级RNN通常预测序列中的下一个字符
        # 对于生成，我们只关心最后一个时间步的输出
        
        # 提取最后一个时间步的输出
        # out 形状: (batch_size, hidden_size)
        out = self.fc(out[:, -1, :]) 
        
        # out 形状: (batch_size, output_size)
        return out, hidden

    def init_hidden(self, batch_size):
        """初始化隐藏状态和细胞状态"""
        # 形状: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

# --- 3. 辅助函数 ---


def sample(preds, temperature=1.0):
    """
    从预测概率张量中采样索引的辅助函数。
    
    参数:
    preds -- 预测概率的PyTorch张量 (形状为 (n_x,))
    temperature -- 温度参数，控制随机性程度
    
    返回:
    采样的字符索引 (整数)
    """
    # 将张量移动到 CPU 并转换为 numpy
    preds_np = preds.cpu().detach().numpy().astype('float64')
    
    # 应用温度参数
    preds_np = np.log(preds_np) / temperature 
    exp_preds = np.exp(preds_np) 
    preds_np = exp_preds / np.sum(exp_preds)  # 重新归一化
    
    # 多项式采样
    probas = np.random.multinomial(1, preds_np, 1) 
    # out = np.random.choice(range(len(preds_np)), p=probas.ravel()) 
    out = np.argmax(probas) # 从 one-hot 结果中获取索引
    return out

def generate_output(model, chars, char_indices, indices_char, Tx):
    """
    生成诗歌的主函数。
    接收用户输入，然后使用模型完成诗歌创作。
    """
    model.eval() # 设置为评估模式
    
    generated = ''
    usr_input = input("请输入你的诗歌开头，莎士比亚机器将完成它。你的输入是: ")
    
    # 用零填充句子到Tx个字符
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input 

    print("\n\n这是你的诗歌: \n")
    print(usr_input, end='', flush=True)
    
    # 生成400个字符
    for i in range(400):
        # 1. 准备预测输入：形状 (1, Tx, len(chars))
        x_pred = torch.zeros(1, Tx, len(chars), dtype=torch.float32).to(device)

        # 2. 将当前句子转换为one-hot编码
        for t, char in enumerate(sentence):
            if char != '0' and char in char_indices: # 忽略填充的零和未知的字符
                x_pred[0, t, char_indices[char]] = 1.0
            
        # 3. 进行预测 (不需要初始化隐藏状态，因为我们在每个时间步都传入整个序列)
        with torch.no_grad():
            # 初始化一个假的隐藏状态，虽然在这个设置中可能不会被模型实际用于生成
            # 如果模型是 stateful 的，则需要维护 hidden
            hidden = model.init_hidden(1) 
            preds_logits, _ = model(x_pred, hidden)
            
            # 使用 softmax 将 logits 转换为概率
            preds = torch.softmax(preds_logits, dim=-1)[0] # 形状 (n_x,)

        # 4. 采样下一个字符
        next_index = sample(preds, temperature=1.2) # 调整温度参数以控制随机性
        next_char = indices_char[next_index]

        # 5. 更新生成的文本和当前句子
        generated += next_char
        sentence = sentence[1:] + next_char  # 滑动窗口

        # 6. 输出下一个字符
        print(next_char, end='', flush=True)

        if next_char == '\n':
            continue
    
    print("\n\n诗歌生成完成！")
    model.train() # 恢复到训练模式

def fix_paths():
    """修复导入路径和文件路径"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)

def main():
    """主函数"""
    fix_paths()
    print("加载文本数据...")
    # 假设 'shakespeare.txt' 在当前运行目录下
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
    n_x = len(chars) # 词汇表大小
    
    print(f"唯一字符数量: {n_x}")
       
    print("定义模型...")
    # 模型参数
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    MODEL_WEIGHT_PATH = 'models/model_shakespeare_best.pth'  # 预训练权重路径
    
    # 实例化模型并移动到设备
    model = ShakespeareModel(n_x, HIDDEN_SIZE, NUM_LAYERS, n_x).to(device)
    # 指定预训练权重文件的路径

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device))
        model.eval()  # 设置为评估模式
        print(f"成功加载预训练模型权重：{MODEL_WEIGHT_PATH}")
    except FileNotFoundError:
        print(f"错误：找不到权重文件 {MODEL_WEIGHT_PATH}。请确保训练脚本已运行，或文件路径正确。")
        return  # 退出程序
    # 调用生成函数
    generate_output(model, chars, char_indices, indices_char, Tx)


# 主程序入口
if __name__ == "__main__":
    main()