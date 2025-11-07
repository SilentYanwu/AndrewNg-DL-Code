# 看看测试集，到底为什么识别不准

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import math

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


# 设置 Matplotlib 使用支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows/Linux
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # MacOS
    except:
        print("未找到中文字体，标题可能显示为方块。")
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_dataset_samples(h5_path, set_name, title, num_samples=25):
    """
    从H5文件中加载数据并绘制一个样本网格。
    """
    if not os.path.exists(h5_path):
        print(f"❌ 错误: 找不到文件 {h5_path}")
        print("请确保 'datasets' 文件夹与此脚本位于同一目录，")
        print("或者 h5_path 变量指向了正确的位置。")
        return

    try:
        # 使用 'with' 语句安全地打开文件
        with h5py.File(h5_path, "r") as f:
            X = np.array(f[f"{set_name}_x"][:])
            Y = np.array(f[f"{set_name}_y"][:])
            
            # 如果有 list_classes，也加载一下
            if "list_classes" in f:
                classes = f["list_classes"][:]
                print(f"在 {h5_path} 中找到类别: {classes}")
            else:
                classes = list(range(int(np.max(Y)) + 1))

    except Exception as e:
        print(f"❌ 读取 H5 文件时出错: {e}")
        return

    print(f"--- 正在显示: {title} ---")
    print(f"图像数据形状 (X): {X.shape}") # (N, 64, 64, 3)
    print(f"标签数据形状 (Y): {Y.shape}") # (N,) or (N, 1)
    
    # 确保 Y 是一维的
    if Y.ndim > 1:
        Y = Y.squeeze()

    # --- 创建网格 ---
    # 计算网格的行数和列数，例如 5x5
    grid_size = math.ceil(math.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    # 从数据集中随机选择索引
    total_images = X.shape[0]
    indices = np.random.choice(total_images, size=num_samples, replace=False)
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            idx = indices[i]
            img = X[idx]
            label = Y[idx]
            
            ax.imshow(img)
            ax.set_title(f"标签 (Label): {label}", fontsize=10)
            ax.axis('off')
        else:
            # 隐藏多余的子图
            ax.axis('off')
            
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    plt.show()

if __name__ == "__main__":
    # 定义 H5 文件路径 (假设 'datasets' 文件夹在同级)
    train_h5_path = os.path.join('datasets', 'train_signs.h5')
    test_h5_path = os.path.join('datasets', 'test_signs.h5')

    # 1. 查看训练集 (train_set)
    # 你的“验证集”是从这个数据集中随机分割出来的，所以看它就够了
    plot_dataset_samples(
        train_h5_path, 
        set_name="train_set", 
        title="训练集 (Train Set) 随机样本"
    )

    # 2. 查看测试集 (test_set)
    plot_dataset_samples(
        test_h5_path, 
        set_name="test_set", 
        title="测试集 (Test Set) 随机样本"
    )