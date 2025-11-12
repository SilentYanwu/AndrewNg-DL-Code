# dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image

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
        """
        self.h5_path = h5_path
        self.set_name = set_name
        self.transform = transform
        self.h5_file = None
        self.X = None
        self.Y = None
        
        # 必须在 __init__ 中获取长度
        with h5py.File(self.h5_path, 'r') as f:
            self.data_len = len(f[f'{set_name}_y'])

    def __len__(self):
        """
        返回数据集大小
        """
        return self.data_len

    def __getitem__(self, idx):
        """
        根据索引获取数据样本
        """
        
        # 4. (核心) 检查这个 worker 是否已经打开了文件
        if self.h5_file is None:
            # 如果没有，就为这个 worker 打开一个 *它自己* 的文件句柄
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.X = self.h5_file[f'{self.set_name}_x']
            self.Y = self.h5_file[f'{self.set_name}_y']

        # 5. 从 worker 自己的句柄中加载数据
        image = self.X[idx]
        label = int(self.Y[idx])

        # 转换为 PIL.Image 以便应用 torchvision 变换
        image = Image.fromarray(image)

        # 应用数据变换
        if self.transform:
            image = self.transform(image)
        
        # 返回处理后的数据和标签
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_classes(self):
        """
        获取所有类别列表
        """
        with h5py.File(self.h5_path, 'r') as f:
            return list(f['list_classes'][:])