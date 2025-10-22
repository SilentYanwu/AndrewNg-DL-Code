# -*- coding: utf-8 -*-
# =========================================================
# 功能: 加载训练好的 cat_model.pth 模型，对单张图片进行猫/非猫预测
# =========================================================

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os,sys

# =========================================================
# 零、路径和字体
# =========================================================
# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)

# 在导入本地之前调用
fix_paths()

# 设置 Matplotlib 使用支持中文的字体（Windows 推荐 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# =========================================================
# 一、设备设置（GPU 优先）
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# =========================================================
# 二、定义模型结构（需与训练时完全一致）
# =========================================================
class LLayerNet(nn.Module):
    def __init__(self, layer_dims, dropout_prob=0.3):
        super(LLayerNet, self).__init__()
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))  # 全连接层
            if i < len(layer_dims) - 1:  # 最后一层不加激活
                layers.append(nn.BatchNorm1d(layer_dims[i]))  # 加 BatchNorm
                layers.append(nn.ReLU())                     # 激活函数
                layers.append(nn.Dropout(p=dropout_prob))     # 加 Dropout
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        out = self.model(x)
        return torch.sigmoid(out)

# 初始化网络（必须与训练文件的结构一致）
layer_dims = [64*64*3, 64, 32, 8, 1]
model = LLayerNet(layer_dims).to(device)

# =========================================================
# 三、加载训练好的模型参数
# =========================================================
model_path = "cat_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ 成功加载模型参数：{model_path}")

# =========================================================
# 四、图片预处理函数（保持与训练时一致）
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

def preprocess_image(image_path):
    """加载并预处理输入图片"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ 无法读取图片: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # 增加 batch 维度
    return image_tensor, image

# =========================================================
# 五、预测函数
# =========================================================
def predict_image(model, image_tensor):
    """输入图像 tensor，输出预测结果（猫/非猫）"""
    with torch.no_grad():
        output = model(image_tensor)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0
    return pred, prob

# =========================================================
# 六、交互式预测
# =========================================================
if __name__ == "__main__":
    while True:
        image_path = input("请输入图片路径（支持 jpg/png/bmp）：")
        try:
            image_tensor, image_show = preprocess_image(image_path)
            pred, prob = predict_image(model, image_tensor)

            # 显示结果
            plt.imshow(image_show)
            plt.axis("off")
            title = f"预测结果: 猫 😺 (置信度 {prob:.3f})" if pred == 1 else f"预测结果: 非猫 😶 (置信度 {1 - prob:.3f})"
            plt.title(title)

            if pred == 1:
                print(f"✅ 模型预测结果：这是一只猫！（置信度 {prob:.3f}）")
            else:
                print(f"❌ 模型预测结果：这不是猫。（置信度 {1 - prob:.3f}）")
            plt.show()

        except Exception as e:
            print(e)

        again = input("是否继续预测其他图片？(y/n): ")
        if again.lower() != "y":
            print("程序结束。")
            break
