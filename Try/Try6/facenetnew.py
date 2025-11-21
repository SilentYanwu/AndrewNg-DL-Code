import torch
import torch.nn as nn

# 这是一个非常流行的 PyTorch FaceNet 库，内置了在 VGGFace2 数据集上训练好的权重
# pip install facenet_pytorch --no-deps 建议不带依赖安装，以避免版本冲突
from facenet_pytorch import InceptionResnetV1

def load_model_wrapper():
    """
    加载预训练好的 FaceNet 模型
    """
    print("正在加载 InceptionResnetV1 (Pretrained on VGGFace2)...")
    
    # pretrained='vggface2': 自动下载并加载在数百万张人脸数据上训练好的权重
    # classify=False: 我们只需要提取特征(Embedding)，不需要分类层
    model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    
    return model