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
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")

    train_x = np.array(train_dataset["train_set_x"][:])  # (209, 64, 64, 3)
    train_y = np.array(train_dataset["train_set_y"][:]).reshape(-1, 1)
    test_x = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:]).reshape(-1, 1)
    return train_x, train_y, test_x, test_y

train_x_orig, train_y, test_x_orig, test_y = load_dataset()

# =========================================================
# 三、定义 Dataset 类
# =========================================================
class CatDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # OpenCV 是 BGR，需要转成 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        # 标签转为 float tensor
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

# =========================================================
# 四、定义数据增强（transforms）
# =========================================================
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
train_images = train_x_orig[:180]
train_labels = train_y[:180]
val_images = train_x_orig[180:]
val_labels = train_y[180:]

train_dataset = CatDataset(train_images, train_labels, transform=train_transform)
val_dataset = CatDataset(val_images, val_labels, transform=val_transform)
test_dataset = CatDataset(test_x_orig, test_y, transform=val_transform)

# 数据加载器（自动打乱和批处理）
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =========================================================
# 六、定义神经网络
# =========================================================
class LLayerNet(nn.Module):
    def __init__(self, layer_dims, dropout_prob=0.3):
        super(LLayerNet, self).__init__()
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))  # 全连接层
            if i < len(layer_dims) - 1:  # 最后一层不加激活
                layers.append(nn.BatchNorm1d(layer_dims[i]))  # 加 BatchNorm
                layers.append(nn.ReLU())                     # 激活函数
                layers.append(nn.Dropout(p=dropout_prob))     # 加 Dropout
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        out = self.model(x)
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
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")

    # 绘制损失曲线
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

train_model(model, train_loader, val_loader, num_epochs=80)

# =========================================================
# 九、测试集准确率
# =========================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc

print(f"测试集准确率: {evaluate(model, test_loader) * 100:.2f}%")

# =========================================================
# 十、保存模型
# =========================================================
# 假设 evaluate(model, test_loader) 
acc = evaluate(model, test_loader) * 100
model_path = f"cat_model_{acc:.2f}%.pth"   
torch.save(model.state_dict(), model_path)
print(f"✅ 模型已保存为 {model_path}")
