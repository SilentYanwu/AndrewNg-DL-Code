'''
PyTorch GPU 版本 - 三层神经网络训练脚本识别手
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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
torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

# =========================================================
# 一、数据加载与预处理
# =========================================================
print("加载数据...")
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

# =========================================================
# 二、转换为PyTorch张量
# =========================================================
# 转置数据以适应PyTorch的格式 (samples, features)
X_train_tensor = torch.FloatTensor(X_train.T).to(device)
X_test_tensor = torch.FloatTensor(X_test.T).to(device)
Y_train_tensor = torch.FloatTensor(Y_train.T).to(device)
Y_test_tensor = torch.FloatTensor(Y_test.T).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================================================
# 三、定义神经网络模型
# =========================================================
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size=12288, hidden1_size=25, hidden2_size=12, output_size=6):
        super(ThreeLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)
        self.relu = nn.ReLU()
        
        # 初始化权重
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # 不使用softmax，因为CrossEntropyLoss自带
        return x

# =========================================================
# 四、实例化模型、损失函数和优化器
# =========================================================
model = ThreeLayerNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("模型结构:")
print(model)

# =========================================================
# 五、训练函数
# =========================================================
def train_model(model, train_loader, test_loader, num_epochs=1500, print_cost=True, is_plot=True):
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_Y in train_loader:
            # 前向传播
            outputs = model(batch_X)
            
            # 计算损失 - 注意：CrossEntropyLoss需要类别索引而不是one-hot
            targets = torch.argmax(batch_Y, dim=1)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # 记录损失
        if epoch % 5 == 0:
            train_losses.append(avg_loss)
        
        # 打印损失
        if print_cost and epoch % 100 == 0:
            print(f"epoch = {epoch}    epoch_loss = {avg_loss:.6f}")
    
    # 绘制损失曲线
    if is_plot:
        plt.plot(np.squeeze(train_losses))
        plt.ylabel('loss')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate = 0.0001")
        plt.show()
    
    return model

# =========================================================
# 六、评估函数
# =========================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            targets = torch.argmax(labels, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    acc = correct / total
    return acc

# =========================================================
# 七、开始训练
# =========================================================
print("开始训练三层神经网络...")
start_time = time.time()

# 训练模型
model = train_model(model, train_loader, test_loader, num_epochs=1500)

end_time = time.time()
print(f"训练时间 = {end_time - start_time:.2f} 秒")

# =========================================================
# 八、评估模型
# =========================================================
train_accuracy = evaluate(model, train_loader)
test_accuracy = evaluate(model, test_loader)

print(f"训练集的准确率: {train_accuracy:.4f}")
print(f"测试集的准确率: {test_accuracy:.4f}")

# =========================================================
# 九、保存模型
# =========================================================
acc = evaluate(model, test_loader) * 100
model_path = f"three_layer_nn_model1_{acc:.2f}%.pth"   
torch.save(model.state_dict(), model_path)
print(f"✅ 模型已保存为 {model_path}")

# 同时保存模型结构和参数
full_model_path = f"three_layer_nn_full_model2_{acc:.2f}%.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': model,
    'input_size': 12288,
    'hidden1_size': 25,
    'hidden2_size': 12,
    'output_size': 6
}, full_model_path)
print(f"✅ 完整模型已保存为 {full_model_path}")