# -*- coding: utf-8 -*-
# =========================================================
# 功能: 使用 PyTorch + torchvision.transforms 实现猫识别（含数据增强、验证集和模型保存）
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import h5py
import cv2
import os,sys
import matplotlib.pyplot as plt

# =========================================================
# 零、路径修复与调整文字：确保脚本在任何位置都能正确运行
# =========================================================
def fix_paths():
    """修复文件路径，保证数据集可以正确读取"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

fix_paths()


# 设置 Matplotlib 使用支持中文的字体（Windows 推荐 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# =========================================================
# 一、GPU 设置
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# =========================================================
# 二、加载 h5 数据集
# =========================================================
def load_dataset():
    """加载猫 vs 非猫数据集"""
    # h5py.File() 用于打开HDF5格式的文件
    # "r" 表示以只读模式打开
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
    # 将数据x读取为numpy数组,再提取训练标签y。
    train_x = np.array(train_dataset["train_set_x"][:])  # (209, 64, 64, 3)
    train_y = np.array(train_dataset["train_set_y"][:]).reshape(-1, 1)
    test_x = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:]).reshape(-1, 1)
    return train_x, train_y, test_x, test_y


# =========================================================
# 三、定义 Dataset 类
# =========================================================
class CatDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images      # 存储图像数据
        self.labels = labels      # 存储对应标签
        self.transform = transform # 数据预处理/增强操作 数据增强的类型
        
    def __len__(self):
        return len(self.images)     # 数据集大小

    def __getitem__(self, idx):
        # 1. 根据索引获取原始图像和标签
        image = self.images[idx]
        label = self.labels[idx]

        # 2. 颜色空间转换：BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # OpenCV默认使用BGR格式，但PyTorch期望RGB格式
        
        # 3. 确保数据类型正确
        image = image.astype(np.uint8)
        # 确保像素值在0-255范围，uint8类型

        # 4. 应用数据变换（数据增强/预处理）
        if self.transform:
            image = self.transform(image)
        # 这里可能包括：归一化、翻转、旋转等操作

        # 5. 标签转换为PyTorch Tensor
        label = torch.tensor(label, dtype=torch.float32)
        # 因为使用BCELoss需要float32类型

        return image, label

# =========================================================
# 四、定义数据增强（transforms）
# =========================================================
# train_transform 是一个 Compose 类的实例对象，它封装了一系列的图像变换操作。
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(15),           # 随机旋转 ±15 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度变化
    transforms.ToTensor()
])

# 验证集和测试集不需要随机增强
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# =========================================================
# 五、划分训练集和验证集
# =========================================================
# 加载数据集
train_x_orig, train_y, test_x_orig, test_y = load_dataset()
# 划分训练集和验证集
train_images = train_x_orig[:180]
train_labels = train_y[:180]
val_images = train_x_orig[180:]
val_labels = train_y[180:]

train_dataset = CatDataset(train_images, train_labels, transform=train_transform)
val_dataset = CatDataset(val_images, val_labels, transform=val_transform)
test_dataset = CatDataset(test_x_orig, test_y, transform=val_transform)

# 数据加载器（自动打乱和批处理）
# 接收 Dataset 对象，返回 DataLoader 对象 只有训练集打乱数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =========================================================
# 六、定义神经网络
# =========================================================
class LLayerNet(nn.Module):
    def __init__(self, layer_dims, dropout_prob=0.3):
        # layer_dims: 各层神经元数量的列表，例如 [12288, 64, 32, 1]
        super(LLayerNet, self).__init__() #明确调用父类的构造函数
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))  # 全连接层
            if i < len(layer_dims) - 1:  # 第一到倒数第二层都要加激活和正则化，最后一层不需要
                layers.append(nn.BatchNorm1d(layer_dims[i]))    # 加 BatchNorm
                layers.append(nn.ReLU())                        # 激活函数
                layers.append(nn.Dropout(p=dropout_prob))       # 加 Dropout
        self.model = nn.Sequential(*layers)                     # nn.Sequential 是一个容器，将多个层组合在一起
        
        # Xavier 初始化
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        out = self.model(x) #  执行前向传播
        return torch.sigmoid(out)

# 初始化网络
layer_dims = [64*64*3, 64, 32, 8, 1]  # 增加一点复杂度
model = LLayerNet(layer_dims).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-4)


# =========================================================
# 七、训练与验证函数
# =========================================================
def train_model(model, train_loader, val_loader, num_epochs=50):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train() # 训练阶段开始
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)               # 前向传播  
            loss = criterion(outputs, labels)   # 计算损失

            optimizer.zero_grad()               # 梯度清零，清空旧梯度
            loss.backward()                     # 反向传播
            optimizer.step()                    # 更新参数
            running_loss += loss.item()

        # 验证阶段（例如平时测试模型效果）
        model.eval() # 开始验证
        val_loss = 0.0
        with torch.no_grad():   # 验证时不需要计算梯度
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)                   # 只是预测
                loss = criterion(outputs, labels)       # 计算损失
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        #  # 每10轮汇报一次成绩
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")

    # 绘制损失曲线（训练与验证）
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.legend()
    plt.title("训练与验证损失变化")
    plt.xlabel("轮次")
    plt.ylabel("Loss")
    plt.show()

# =========================================================
# 八、训练模型
# =========================================================
Numepochs = 80
train_model(model, train_loader, val_loader, Numepochs)
# =========================================================
# 九、测试集准确率
# =========================================================
def evaluate(model, loader):
    model.eval() # 设置为评估模式，最后测试模型效果
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)          # 前向传播得到预测概率
             # 将概率转换为0/1预测
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    # 计算准确率
    acc = correct / total
    return acc

print(f"测试集准确率: {evaluate(model, test_loader) * 100:.2f}%")

# =========================================================
# 十、保存模型
# =========================================================
# 假设 evaluate(model, test_loader) 
accc = evaluate(model, test_loader) * 100
model_path = f"cat_model_{accc:.2f}%.pth"   
torch.save(model.state_dict(), model_path)
print(f"✅ 模型已保存为 {model_path}")
