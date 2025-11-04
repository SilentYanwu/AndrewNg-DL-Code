# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import argparse
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

# 设置 Matplotlib 使用支持中文的字体（Windows 推荐 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 协调：从 model.py 导入共享模型
from model import SignCNN
# 现代：从 dataset.py 导入数据加载器
from dataset import SignLanguageDataset

# 定义图像尺寸和标准化（保持与原数据一致）
IMG_SIZE = 64

def get_data_loaders(data_dir, batch_size):
    """
    强大：准备数据加载器，包含数据增强和训练/验证集划分
    """
    
    # 强大：为训练集定义数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), # 自动将 [0, 255] PIL 图像转为 [0, 1] Tensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 可选
    ])
    
    # 验证集和测试集不需要增强，只需转为 Tensor
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载完整训练集
    full_train_dataset = SignLanguageDataset(
        os.path.join(data_dir, 'train_signs.h5'), 
        set_name='train_set', 
        transform=train_transform
    )
    
    # 强大：划分训练集和验证集 (e.g., 90% 训练, 10% 验证)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # 注意：验证集应该使用 val_test_transform
    # PyTorch 的 random_split 共享底层数据，但我们可以通过 .dataset 属性访问
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
    
    print(f"总训练样本: {len(full_train_dataset)} -> 拆分为:")
    print(f"  - 训练集: {len(train_dataset)}")
    print(f"  - 验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    model = SignCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("🚀 开始训练...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # 强大：只保存在验证集上表现最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  🎉 新的最佳模型! 准确率: {best_val_acc:.2f}%. 已保存到 {save_path}")
            
    print("✅ 训练完成。")

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.show()
    
    # 在测试集上评估最终的最佳模型
    print("🧪 正在用测试集评估最佳模型...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt")))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"✅ 最终测试集准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练手语 CNN 模型")
    parser.add_argument('--data_dir', type=str, default='datasets', help='H5 数据集所在文件夹')
    parser.add_argument('--save_dir', type=str, default='runs', help='模型保存文件夹')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    
    args = parser.parse_args()
    main(args)