# train_resnet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# 导入本地模块
from dataset import SignLanguageDataset
from resnet_model import create_resnet50

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图像尺寸和标准化（使用 ImageNet 均值和标准差，因为我们用预训练模型）
IMG_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_data_loaders(data_dir, batch_size):
    """
    准备数据加载器，包含数据增强和训练/验证集划分
    """
    
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # 添加更强的增强
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) # 关键：使用 ImageNet 归一化
    ])

    # 验证集和测试集不需要增强
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) # 关键：使用 ImageNet 归一化
    ])
    
    # 加载完整训练集 (使用增强)
    full_train_dataset = SignLanguageDataset(
        os.path.join(data_dir, 'train_signs.h5'), 
        set_name='train_set', 
        transform=train_transform
    )
    
    # 划分训练集和验证集 (90% 训练, 10% 验证)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # 使用固定的随机种子，确保每次划分一致
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # ！！重要！！ 将验证集的数据变换替换为 *不增强* 的版本
    # random_split 后的 val_dataset 仍然指向 full_train_dataset
    # 我们需要一个技巧来修改它的 transform
    # 最简单的方法是重新创建一次 val_dataset，但这会重复加载数据
    # 更高效的方式是创建一个包装器或修改 val_dataset.dataset
    # 为了简单和安全，我们这里重新加载一次（虽然效率稍低）
    
    # 我们创建一个新的实例用于验证，仅为了应用正确的 transform
    # 注意：这假设 val_dataset.indices 存储了正确的索引
    # 一个更干净的方法是让 SignLanguageDataset 接受一个索引列表
    # 但我们保持您原有的 dataset.py 不变，采用以下策略：
    
    # 策略调整：在 random_split 之后，我们修改 val_dataset 的 transform
    # 幸运的是，PyTorch 的 Subset 对象允许我们访问其底层的 dataset
    # 我们可以 *临时* 修改底层 dataset 的 transform
    # 但这会带来风险，因为 train_dataset 也共享它
    
    # 最安全、最清晰的策略：
    # 1. 加载两次 train_signs.h5
    dataset_for_train = SignLanguageDataset(
        os.path.join(data_dir, 'train_signs.h5'), 
        set_name='train_set', 
        transform=train_transform
    )
    dataset_for_val = SignLanguageDataset(
        os.path.join(data_dir, 'train_signs.h5'), 
        set_name='train_set', 
        transform=val_test_transform # 验证集使用 *无增强* 变换
    )
    
    # 2. 使用相同的种子和索引进行划分
    indices = torch.randperm(len(dataset_for_train), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset_for_train, train_indices)
    val_dataset = torch.utils.data.Subset(dataset_for_val, val_indices)


    # 加载测试集
    test_dataset = SignLanguageDataset(
        os.path.join(data_dir, 'test_signs.h5'), 
        set_name='test_set', 
        transform=val_test_transform
    )

    # 创建 DataLoaders
    # 使用 num_workers > 0 来利用您安全的 dataset.py
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"总训练样本: {len(full_train_dataset)} -> 拆分为:")
    print(f"   - 训练集: {len(train_dataset)}")
    print(f"   - 验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    训练一个epoch (支持混合精度)
    """
    model.train()
    running_loss = 0.0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        # 缩放梯度
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    """
    在验证集或测试集上评估模型
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 评估时也使用混合精度
            with torch.cuda.amp.autocast():
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
    """主训练函数"""
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 正在使用设备: {device}")
    
    # 2. 创建模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 3. 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    # 4. 初始化模型
    #    (创建预训练、冻结卷积层的模型)
    model = create_resnet50(
        num_classes=6, 
        use_pretrained=True, 
        freeze_layers=True  # 先冻结训练
    ).to(device)
    
    # 5. 定义损失函数、优化器、学习率调度器
    criterion = nn.CrossEntropyLoss()
    
    # 仅优化解冻的参数 (这里是 model.fc)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # 6. 初始化混合精度 (AMP) 缩放器
    scaler = torch.amp.GradScaler()
    
    # 7. 训练循环
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    # 迁移学习阶段的代码
    print("\n--- 阶段 1: 训练分类头 (冻结卷积层) ---")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  🎉 新的最佳模型! 准确率: {best_val_acc:.2f}%. 已保存到 {save_path}")

    # 8. (可选但推荐) 解冻模型并进行端到端微调
    print("\n--- 阶段 2: 解冻并微调 (End-to-End Fine-tuning) ---")
    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True
        
    # 为解冻后的模型创建一个新的优化器，使用更小的学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr / 10) # 使用 1/10 的学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # 再训练几个 epochs
    fine_tune_epochs = args.epochs // 2 # 比如再训练 1/2 的轮数
    
    for epoch in range(fine_tune_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # 记录到 history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Fine-tune Epoch {epoch+1:02d}/{fine_tune_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  🎉 新的最佳模型 (Fine-tuned)! 准确率: {best_val_acc:.2f}%. 已保存到 {save_path}")

    print("✅ 训练完成。")
    
    # 9. 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.axvline(x=args.epochs-1, color='gray', linestyle='--', label='Fine-tune 开始')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("损失曲线")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='验证准确率', color='orange')
    plt.axvline(x=args.epochs-1, color='gray', linestyle='--', label='Fine-tune 开始')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title("验证准确率曲线")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curves.png"))
    print(f"📊 训练曲线已保存到 {args.save_dir}/training_curves.png")
    plt.show()
    
    # 10. 在测试集上评估最终的最佳模型
    print("🧪 正在用测试集评估最佳模型...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt")))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"✅ 最终测试集准确率: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练手语 ResNet-50 模型")
    parser.add_argument('--data_dir', type=str, default='datasets', help='H5 数据集所在文件夹')
    parser.add_argument('--save_dir', type=str, default='runs', help='模型保存文件夹')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率 (用于训练分类头)')
    parser.add_argument('--epochs', type=int, default=20, help='*初始*训练轮数 (仅分类头)')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    
    args = parser.parse_args()
    main(args)