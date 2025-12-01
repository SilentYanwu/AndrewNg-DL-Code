'''
trainmodel.py
莎士比亚诗歌生成器 - 训练脚本 (PyTorch)
功能：读取shakespeare.txt，训练LSTM模型，并保存最佳权重和最新权重供生成器使用。
'''
import os,sys
import io
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    # 切换到当前文件所在目录
    os.chdir(current_dir)

# --- 1. 配置参数 ---
FILE_PATH = 'shakespeare.txt'
MODEL_SAVE_PATH = 'models'          # 模型保存目录
SEQ_LENGTH = 40         # 序列长度 (Tx)
STRIDE = 3              # 滑动窗口步长
BATCH_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2          # LSTM层数
DROPOUT = 0.2           # Dropout比率
LEARNING_RATE = 0.001   # 学习率
EPOCHS = 100            # 训练轮数

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的训练设备: {device}")

# 确保模型保存目录存在
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# --- 2. 模型定义 ---
class ShakespeareModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        '''
            参数介绍：input_size: 输入特征维度，output_size: 输出特征维度
            hidden_size: 隐藏层特征维度，num_layers: LSTM层数
            dropout_rate: Dropout比率
        '''
        super(ShakespeareModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步的输出用于预测下一个字符
        # out[:, -1, :] shape: (batch, hidden_size)
        out = self.fc(out[:, -1, :]) 
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

# --- 3. 数据集类 ---
class ShakespeareDataset(Dataset):
    def __init__(self, text, char_indices, seq_length, stride):
        '''
        参数:
            text -- 完整文本字符串
            char_indices -- 字符到索引的映射字典
            seq_length -- 序列长度 (Tx)
            stride -- 滑动窗口步长
        '''
        self.text = text
        self.char_indices = char_indices
        self.vocab_size = len(char_indices)
        self.seq_length = seq_length
        
        # 预先生成索引序列，而不是存储巨大的one-hot矩阵
        self.X_indices = []
        self.Y_indices = []
        
        print("正在构建数据集索引...")
        for i in range(0, len(text) - seq_length, stride):
            # 存储字符对应的整数索引
            window = text[i: i + seq_length]
            next_char = text[i + seq_length]
            
            self.X_indices.append([char_indices[c] for c in window])
            self.Y_indices.append(char_indices[next_char])
            
        print(f"样本总数: {len(self.X_indices)}")

    def __len__(self):
        return len(self.X_indices)

    def __getitem__(self, idx):
        # 获取索引序列
        x_seq = self.X_indices[idx]
        y_idx = self.Y_indices[idx]
        
        # 实时生成 One-Hot 向量 (Tx, vocab_size)
        x_tensor = torch.zeros(self.seq_length, self.vocab_size, dtype=torch.float32)
        for t, char_idx in enumerate(x_seq):
            x_tensor[t, char_idx] = 1.0
            
        # y 直接返回类别索引 (CrossEntropyLoss 需要 LongTensor)
        y_tensor = torch.tensor(y_idx, dtype=torch.long)
        
        return x_tensor, y_tensor


# --- 4. 主训练流程 (已集成最佳模型保存逻辑) ---
def train():
    # 1. 读取数据
    print(f"正在读取 {FILE_PATH}...")
    try:
        text = io.open(FILE_PATH, encoding='utf-8').read().lower()
    except FileNotFoundError:
        print(f"错误：找不到 {FILE_PATH}，请确保文件在同目录下。")
        return

    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    vocab_size = len(chars)
    print(f"词汇表大小: {vocab_size}")

    # 2. 准备 DataLoader
    dataset = ShakespeareDataset(text, char_indices, SEQ_LENGTH, STRIDE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. 初始化模型
    model = ShakespeareModel(vocab_size, HIDDEN_SIZE, NUM_LAYERS, vocab_size, DROPOUT).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE) 

    print("\n开始训练...")
    start_time = time.time()
    
    # 初始化最佳 Loss 为无穷大，用于保存最佳模型
    best_loss = float('inf') 
    best_epoch = -1

    for epoch in range(EPOCHS):
        epoch_loss = 0
        model.train()
        
        # 批次循环
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 初始化隐藏状态 (每个batch重置，因为我们不是stateful RNN)
            hidden = model.init_hidden(BATCH_SIZE)
            
            # 前向传播
            # 注意：LSTM需要detach隐藏状态，否则计算图会无限累积
            hidden = tuple([h.data for h in hidden])
            # 梯度清零
            optimizer.zero_grad()

            # 前向
            output, hidden = model(x_batch, hidden)
            # 计算损失
            loss = criterion(output, y_batch)
            # 反向传播
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        # 计算平均Loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} 完成. 平均 Loss: {avg_loss:.4f}")
        
         # 1. 始终保存最新模型
        last_path = os.path.join(MODEL_SAVE_PATH, "model_shakespeare_last.pth")
        torch.save(model.state_dict(), last_path)

        # 2. 如果当前 Loss 是历史最低，则保存为最佳模型
        if avg_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = avg_loss
            best_path = os.path.join(MODEL_SAVE_PATH, "model_shakespeare_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"★ 发现新最佳 Loss: {best_loss:.4f}，已保存最佳模型！")
        else:
            print(f"当前 Loss ({avg_loss:.4f}) 未低于最佳 ({best_loss:.4f})，未更新最佳模型。")

        print(f"模型已保存 (最新版: {last_path})\n")
        
    total_time = time.time() - start_time
    print(f"训练完成！总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳模型出现在 Epoch {best_epoch}，Loss: {best_loss:.4f}")

if __name__ == "__main__":
    fix_paths()
    train()
    print("训练脚本执行完毕。")