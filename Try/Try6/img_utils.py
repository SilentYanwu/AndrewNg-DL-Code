import torch
import numpy as np
import cv2
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

def img_to_encoding(image_path, model, device):
    """
    加载图像 -> 转为 Tensor -> 归一化 -> 模型推理 -> 返回 Embedding
    
    Args:
        image_path: 图片路径
        model: 已加载的 PyTorch 模型
        device: 'cuda' or 'cpu'
    Returns:
        embedding: numpy array (128,)
    """
    # 1. 读取图片 (BGR)
    img1 = cv2.imread(image_path, 1)
    if img1 is None:
        raise FileNotFoundError(f"无法找到图片: {image_path}")

    # 2. BGR -> RGB
    img = img1[..., ::-1]

    # 3. Resize 到模型需要的尺寸 (96x96)
    img = cv2.resize(img, (96, 96))

    # 4. 归一化 (0-255 -> 0.0-1.0) 并转置 (H,W,C) -> (C,H,W)
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    
    # 5. 增加 Batch 维度 (1, 3, 96, 96) 并转为 FloatTensor
    x_train = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    
    # 6. 移至 GPU
    x_train = x_train.to(device)

    # 7. 推理 (No Grad 模式)
    model.eval()
    with torch.no_grad():
        embedding = model(x_train)
    
    # 8. 返回 CPU numpy 数组
    return embedding.cpu().numpy()

def load_database(database_path, model, device):
    """
    加载整个数据库的图片并计算编码
    """
    database = {}
    # 遍历目录下的所有图片
    # 假设 database_path 是一个字典 {name: path} 或者文件夹路径
    # 这里保持原代码逻辑，手动构建字典
    return database