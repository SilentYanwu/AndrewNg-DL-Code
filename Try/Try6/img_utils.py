# img_utils.py
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
    加载图像 -> Resize -> 标准化 -> 模型推理 -> 512维向量
    """
    # 1. 读取图片
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法找到图片: {image_path}")

    # 2. BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. [关键修改] Resize 到 160x160
    # facenet-pytorch 的预训练模型是在 160x160 分辨率下训练的
    img_rgb = cv2.resize(img_rgb, (160, 160))

    # 4. [关键修改] 标准化 (Whitening)
    # 预训练模型要求的标准化方式：(像素值 - 127.5) / 128.0
    # 这种方式将像素归一化到 [-1, 1] 之间，而不是之前的 [0, 1]
    img_tensor = torch.tensor(img_rgb).float()
    img_tensor = (img_tensor - 127.5) / 128.0

    # 5. 转换维度 (H,W,C) -> (C,H,W) 并增加 Batch 维度
    # (160, 160, 3) -> (3, 160, 160) -> (1, 3, 160, 160)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # 6. GPU 推理
    img_tensor = img_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        # 输出维度通常是 (1, 512)
        embedding = model(img_tensor)
    
    # 返回 CPU numpy 数组
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