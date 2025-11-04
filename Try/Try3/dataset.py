# dataset.py (已修复)

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image # 转换是 torchvision.transforms 所必需的

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
        Args:
            h5_path (str): .h5 文件的路径
            set_name (str): 'train_set' 或 'test_set'
            transform (callable, optional): 应用于样本的变换
        """
        # 1. (安全) 只存储路径，不打开文件
        self.h5_path = h5_path
        self.set_name = set_name
        self.transform = transform

        # 2. (安全) 将文件句柄和数据集指针初始化为 None
        # 它们将在 __getitem__ 中被 worker 进程填充
        self.h5_file = None
        self.X = None
        self.Y = None
        
        # 3. (安全) 必须在 __init__ 中获取长度。
        #    我们使用 'with' 语句临时打开文件，只为获取长度，然后立即关闭。
        #    这样 h5py.File 对象就不会被存储在 self 中，可以被 pickle。
        with h5py.File(self.h5_path, 'r') as f:
            self.data_len = len(f[f'{set_name}_y'])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        
        # 4. (核心) 这是在 worker 进程中运行的 (如果 num_workers > 0)
        #    检查这个 worker 是否已经打开了文件
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

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_classes(self):
        # 同样，使用 'with' 临时打开，确保安全
        with h5py.File(self.h5_path, 'r') as f:
            return list(f['list_classes'][:])

if __name__ == "__main__":
    # 测试数据集加载
    from torchvision import transforms
    
    # 定义一个简单的变换：转为 Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dset = SignLanguageDataset(
        'datasets/train_signs.h5', 
        set_name='train_set', 
        transform=transform
    )
    
    img, label = train_dset[0]
    print(f"加载样本 0, 标签: {label}")
    print(f"图像张量形状: {img.shape}, 类型: {img.dtype}") # [3, 64, 64]
    
    # 测试多进程加载 (需要 train.py)
    print("\n尝试使用 DataLoader (num_workers=2)...")
    try:
        loader = torch.utils.data.DataLoader(train_dset, batch_size=2, num_workers=2)
        for i, (img_batch, label_batch) in enumerate(loader):
            print(f"Batch {i}: {img_batch.shape}, {label_batch.shape}")
            if i > 2:
                break
        print("✅ 多进程加载测试成功！")
    except Exception as e:
        print(f"❌ 多进程加载测试失败: {e}")