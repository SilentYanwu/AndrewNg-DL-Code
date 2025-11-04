# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import argparse
import os, sys

# 路径修复函数 - 确保模块导入和文件路径正确
def fix_paths():
    """修复导入路径和文件路径，确保脚本在不同环境下都能正常运行"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)

# 在导入本地文件/模型之前调用路径修复
fix_paths()

# 设置 Matplotlib 使用支持中文的字体（Windows 推荐 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 从 model.py 导入共享的CNN模型
from model import SignCNN
# 从 dataset.py 导入数据加载器
from dataset import SignLanguageDataset

# 定义图像尺寸和标准化参数（保持与原数据一致）
IMG_SIZE = 64

def get_data_loaders(data_dir, batch_size):
    """
    准备数据加载器，包含数据增强和训练/验证集划分
    
    参数:
        data_dir: 数据目录路径
        batch_size: 批处理大小
        
    返回:
        train_loader, val_loader, test_loader: 训练、验证、测试数据加载器
    """
    
    # 为训练集定义数据增强策略，提高模型泛化能力
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图像尺寸
        transforms.RandomHorizontalFlip(),        # 随机水平翻转
        transforms.RandomRotation(15),            # 随机旋转 ±15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),                    # 将PIL图像转为Tensor [0,1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 可选标准化
    ])
    
    # 验证集和测试集不需要增强，只需基础预处理
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图像尺寸
        transforms.ToTensor(),                    # 将PIL图像转为Tensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 可选标准化
    ])
    
    # 加载完整训练集
    full_train_dataset = SignLanguageDataset(
        os.path.join(data_dir, 'train_signs.h5'), 
        set_name='train_set', 
        transform=train_transform
    )
    
    # 划分训练集和验证集 (90% 训练, 10% 验证)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # 注意：验证集应该使用val_test_transform
    # 由于random_split共享底层数据，我们需要通过.dataset属性访问并修改transform
    val_dataset.dataset.transform = val_test_transform 

    # 加载测试集
    test_dataset = SignLanguageDataset(
        os.path.join(data_dir, 'test_signs.h5'), 
        set_name='test_set', 
        transform=val_test_transform
    )

    # 创建 DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 打印数据集统计信息
    print(f"总训练样本: {len(full_train_dataset)} -> 拆分为:")
    print(f"  - 训练集: {len(train_dataset)}")
    print(f"  - 验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    训练一个epoch
    
    参数:
        model: 神经网络模型
        loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备 (CPU/GPU)
        
    返回:
        average_loss: 该epoch的平均损失
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    
    # 遍历训练数据批次
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # 返回该epoch的平均损失
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    """
    在验证集或测试集上评估模型
    
    参数:
        model: 神经网络模型
        loader: 验证/测试数据加载器
        criterion: 损失函数
        device: 训练设备 (CPU/GPU)
        
    返回:
        avg_loss: 平均损失
        accuracy: 准确率百分比
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 在评估模式下不计算梯度
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 累计损失
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    # 计算平均损失和准确率
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main(args):
    """主训练函数"""
    # 设置训练设备 (优先使用GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    # 初始化模型、损失函数和优化器
    model = SignCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam优化器
    
    # 记录训练历史
    best_val_acc = 0.0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': []
    }
    
    print("🚀 开始训练...")
    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 在验证集上评估
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印训练进度
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        # 保存验证集上表现最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  🎉 新的最佳模型! 准确率: {best_val_acc:.2f}%. 已保存到 {save_path}")
            
    print("✅ 训练完成。")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("损失曲线")
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='验证准确率', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title("验证准确率曲线")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 在测试集上评估最终的最佳模型
    print("🧪 正在用测试集评估最佳模型...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt")))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"✅ 最终测试集准确率: {test_acc:.2f}%")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="训练手语 CNN 模型")
    parser.add_argument('--data_dir', type=str, default='datasets',help='H5 数据集所在文件夹')
    parser.add_argument('--save_dir', type=str, default='runs',help='模型保存文件夹')
    parser.add_argument('--lr', type=float, default=1e-3,help='学习率')
    parser.add_argument('--epochs', type=int, default=20,help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,help='批处理大小')
    
    # 解析参数并启动训练
    args = parser.parse_args()
    main(args)