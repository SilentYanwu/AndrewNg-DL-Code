# dataset.py
import torch
from torch.utils.data import Dataset  
import h5py  # 用于读取 HDF5 格式的文件
import numpy as np
from PIL import Image  # 图像处理库，转换是 torchvision.transforms 所必需的

"""
📚 Python 类基础知识回顾：

class ClassName(ParentClass):
    def __init__(self, parameters):  # 构造函数，创建对象时调用
        self.attribute = value       # 实例属性，每个对象独有的
    
    def method(self, parameters):    # 实例方法
        return result

Dataset 类是 PyTorch 的数据集基类，自定义数据集必须继承它
并实现三个方法：__init__、__len__、__getitem__
"""

class SignLanguageDataset(Dataset):
    """
    用于 signs.h5 数据的 PyTorch 自定义数据集
    
    [已修复]：
    这个版本是多进程安全的 (num_workers > 0)。
    它不在 __init__ 中打开 h5_file，而是在 __getitem__ 中
    为每个 worker 单独打开文件句柄。
    """
    
    def __init__(self, h5_path, set_name='train_set', transform=None):
        """
        构造函数：初始化数据集对象
        
        🎯 比喻：就像建立一个图书馆的借书系统
        - 我们记录书库的位置，但不立即打开所有书架
        - 等读者来借书时，再打开对应的书架
        
        Args:
            h5_path (str): .h5 文件的路径 - 就像书库的地址
            set_name (str): 'train_set' 或 'test_set' - 就像选择"训练区"或"测试区"
            transform (callable, optional): 应用于样本的变换 - 就像借书前的消毒处理
        """
        # 1. (安全) 只存储路径，不打开文件
        # 🗺️ 比喻：只记录"书库在哪"，不打开书库大门
        self.h5_path = h5_path        # 书库位置
        self.set_name = set_name      # 区域选择
        self.transform = transform    # 数据处理流程

        # 2. (安全) 将文件句柄和数据集指针初始化为 None
        # 🔑 比喻：钥匙串目前是空的，等需要时再配钥匙
        # 它们将在 __getitem__ 中被 worker 进程填充
        self.h5_file = None  # 文件句柄 - 就像书库的钥匙
        self.X = None        # 图像数据指针 - 就像图像书架的编号
        self.Y = None        # 标签数据指针 - 就像标签书架的编号
        
        # 3. (安全) 必须在 __init__ 中获取长度
        # 📏 比喻：先数一下书库里总共有多少本书
        # 使用 'with' 语句临时打开文件，只为获取长度，然后立即关闭
        # 这样 h5py.File 对象就不会被存储在 self 中，可以被 pickle
        with h5py.File(self.h5_path, 'r') as f:  # 临时打开书库数一下
            self.data_len = len(f[f'{set_name}_y'])  # 统计这个区域有多少本书

    def __len__(self):
        """
        返回数据集大小
        
        📚 比喻：告诉别人这个书库有多少本书
        """
        return self.data_len

    def __getitem__(self, idx):
        """
        根据索引获取数据样本
        
        🎯 比喻：读者根据书号来借书的过程
        - 如果是新读者，先给他配一把钥匙
        - 用钥匙打开书库，找到对应编号的书
        - 对书进行必要的处理（消毒、包装等）
        - 把书交给读者
        
        Args:
            idx: 数据索引，就像书的编号
            
        Returns:
            image: 处理后的图像
            label: 对应的标签
        """
        
        # 4. (核心) 检查这个 worker 是否已经打开了文件
        # 🔑 比喻：检查这个读者是否已经有书库钥匙了
        if self.h5_file is None:
            # 如果没有，就为这个 worker 打开一个 *它自己* 的文件句柄
            # 🗝️ 比喻：给这个读者配一把属于他自己的书库钥匙
            self.h5_file = h5py.File(self.h5_path, 'r')  # 配钥匙，开门
            self.X = self.h5_file[f'{self.set_name}_x']  # 找到图像书架
            self.Y = self.h5_file[f'{self.set_name}_y']  # 找到标签书架

        # 5. 从 worker 自己的句柄中加载数据
        # 📖 比喻：用读者自己的钥匙打开书库，取出对应编号的书
        image = self.X[idx]      # 取出第 idx 本图像书
        label = int(self.Y[idx]) # 取出第 idx 本标签书，并转换为整数

        # 转换为 PIL.Image 以便应用 torchvision 变换
        # 🖼️ 比喻：把书从仓库格式转换成阅读格式
        image = Image.fromarray(image)

        # 应用数据变换（如果有的话）
        # ✨ 比喻：对书进行消毒、包装等处理
        if self.transform:
            image = self.transform(image)
        
        # 返回处理后的数据和标签
        # 📦 比喻：把处理好的书交给读者
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_classes(self):
        """
        获取所有类别列表
        
        📋 比喻：获取书库中所有图书分类的目录
        """
        # 同样，使用 'with' 临时打开，确保安全
        with h5py.File(self.h5_path, 'r') as f:  # 临时打开书库
            return list(f['list_classes'][:])    # 取出分类目录

"""
🔑 关于"句柄"的深入比喻：

想象一个大型图书馆：
- 文件句柄 (self.h5_file) = 书库的钥匙
- 数据集指针 (self.X, self.Y) = 具体书架的编号

❌ 错误做法：
- 馆长只有一把钥匙，让所有工作人员共用
- 结果：工作人员抢钥匙，混乱不堪

✅ 正确做法：
- 每个工作人员配一把自己的钥匙
- 结果：大家各自有序工作，互不干扰

这就是"多进程安全"的精髓！
"""

if __name__ == "__main__":
    """
    测试代码：验证数据集是否能正常工作
    """
    # 测试数据集加载
    from torchvision import transforms
    
    # 定义一个简单的变换：转为 Tensor
    # 🔄 比喻：定义借书时的标准处理流程
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 创建数据集实例
    # 🏢 比喻：建立一个借书系统
    train_dset = SignLanguageDataset(
        'datasets/train_signs.h5', 
        set_name='train_set', 
        transform=transform
    )
    
    # 测试单样本加载
    # 📖 比喻：单个读者借一本书
    img, label = train_dset[0]
    print(f"加载样本 0, 标签: {label}")
    print(f"图像张量形状: {img.shape}, 类型: {img.dtype}") # [3, 64, 64]
    
    # 测试多进程加载
    # 👥 比喻：多个读者同时借书
    print("\n尝试使用 DataLoader (num_workers=2)...")
    try:
        # 创建数据加载器，使用2个工作进程
        loader = torch.utils.data.DataLoader(train_dset, batch_size=2, num_workers=2)
        
        # 批量加载数据
        for i, (img_batch, label_batch) in enumerate(loader):
            print(f"Batch {i}: {img_batch.shape}, {label_batch.shape}")
            if i > 2:  # 只测试前3个批次
                break
        print("✅ 多进程加载测试成功！")
    except Exception as e:
        print(f"❌ 多进程加载测试失败: {e}")