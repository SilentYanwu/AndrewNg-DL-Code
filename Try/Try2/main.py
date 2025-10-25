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
import time,h5py

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

# 设置随机种子
torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

# =========================================================
# 工具函数
# =========================================================
def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# =========================================================
# 一、数据加载与预处理
# =========================================================
print("加载数据...")
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

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

# 转换标签为一维数组
Y_train = Y_train_orig.flatten()
Y_test = Y_test_orig.flatten()

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
Y_train_tensor = torch.LongTensor(Y_train).to(device)
Y_test_tensor = torch.LongTensor(Y_test).to(device)


# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================================================
# 三、定义神经网络模型（使用DNN）
# =========================================================
class LLayerNet(nn.Module):
    def __init__(self, layer_dims, dropout_prob=0.3):
        super(LLayerNet, self).__init__()
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i < len(layer_dims) - 1:
                layers.append(nn.BatchNorm1d(layer_dims[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_prob))
        self.model = nn.Sequential(*layers)

        # Xavier 初始化
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)  # 输出 logits（不加 sigmoid/softmax）



# =========================================================
# 四、实例化模型、损失函数和优化器
# =========================================================
Layer_dims = [12288, 64, 20, 6]  # 输入层、隐藏层1、隐藏层2、输出层
learning_rate = 0.001 # 学习率
L2loss = 0.0001 # 权重衰减（L2正则化）
model = LLayerNet(Layer_dims).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2loss)

print("模型结构:")
print(model)

# =========================================================
# 五、训练函数
# =========================================================
def train_model(model, train_loader, test_loader, num_epochs=500, print_cost=True, is_plot=True):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_X, batch_Y in train_loader:
            # 前向传播
            outputs = model(batch_X)

            # 计算损失 (注意: batch_Y 已是类别索引，不需要 argmax)
            loss = criterion(outputs, batch_Y)

            # 反向传播与优化
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
        plt.xlabel('iterations (per 5 epochs)')
        plt.title("Training Loss Curve")
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
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# =========================================================
# 七、开始训练
# =========================================================
print("开始训练三层神经网络...")
start_time = time.time()

# 训练模型
epochs = 1000 # 训练轮数
model = train_model(model, train_loader, test_loader, epochs)

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
model_path = f"three_layer_nn_model_{acc:.2f}%.pth"   
torch.save(model.state_dict(), model_path)
print(f"✅ 模型已保存为 {model_path}")

# 同时保存模型结构和参数
full_model_path = f"three_layer_nn_full_model_{acc:.2f}%.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': model,
    'input_size': Layer_dims[0],
    'hidden1_size': Layer_dims[1],
    'hidden2_size': Layer_dims[2],
    'output_size': Layer_dims[3]
}, full_model_path)
print(f"✅ 完整模型已保存为 {full_model_path}")