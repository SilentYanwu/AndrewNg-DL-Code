# predict.py
import torch
import cv2
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

# 协调：从 model.py 导入共享模型
from model import SignCNN

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 最佳实践：使用模型训练时的尺寸进行推理
INFER_SIZE = 64 

# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)


def load_model(model_path, device):
    """协调：加载与训练时结构一致的模型"""
    model = SignCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path_or_array):
    """
    更强大的预处理：处理路径或已加载的 numpy 数组
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path_or_array}")
    else:
        # 假设是 BGR numpy 数组
        image = image_path_or_array

    # 1. BGR -> RGB
    if len(image.shape) == 2: # 灰度图
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # 2. 转换为 PIL Image
    image_pil = Image.fromarray(image_rgb)
    
    # 3. 应用与验证集/测试集 *完全相同* 的变换
    preprocess_transform = transforms.Compose([
        transforms.Resize((INFER_SIZE, INFER_SIZE)), # 调整到模型熟悉的尺寸
        transforms.ToTensor(),
        # 如果训练时用了 Normalize, 这里也要用
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 4. 转换为 Tensor 并增加 batch 维度 [C, H, W] -> [1, C, H, W]
    image_tensor = preprocess_transform(image_pil).unsqueeze(0)
    
    # 返回 tensor 用于模型输入, 返回 rgb 图像用于显示
    return image_tensor, image_rgb


def predict_and_show(model, image_path, device, classes):
    """
    执行单张图片预测并显示结果
    """
    try:
        tensor, original_rgb = preprocess_image(image_path)
        tensor = tensor.to(device)
    except ValueError as e:
        print(e)
        return

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
        label = classes[pred.item()]
        conf_value = conf.item() * 100

    print(f"文件: {os.path.basename(image_path)}")
    print(f"  -> 预测类别: {label} | 置信度: {conf_value:.2f}%")

    # === 使用 Matplotlib 显示 (比 OpenCV 窗口更友好) ===
    plt.figure(figsize=(6, 6))
    plt.imshow(original_rgb) # 显示预处理前的原始 RGB 图像
    
    title_text = f"预测结果: {label} (置信度: {conf_value:.2f}%)"
    plt.title(title_text, fontsize=14)
    plt.axis("off")
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = [0, 1, 2, 3, 4, 5] # 假设类别
    
    print(f"正在从 {args.model} 加载模型...")
    model = load_model(args.model, device)
    
    if os.path.isfile(args.input):
        print("--- 单张图片预测 ---")
        predict_and_show(model, args.input, device, classes)
        
    elif os.path.isdir(args.input):
        print("--- 批量文件夹预测 ---")
        for f in os.listdir(args.input):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(args.input, f)
                # 批量预测时只打印结果，不显示图片
                try:
                    tensor, _ = preprocess_image(img_path)
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        outputs = model(tensor)
                        _, pred = torch.max(outputs, 1)
                        print(f"文件: {f} -> 预测: {classes[pred.item()]}")
                except Exception as e:
                    print(f"跳过 {f}: {e}")
    else:
        print(f"❌ 错误: 输入路径无效: {args.input}")


if __name__ == "__main__":
    fix_paths()  # 在导入本地文件
    parser = argparse.ArgumentParser(description="预测手语 CNN 模型")
    parser.add_argument('-m', '--model', type=str, default='runs/best_model.pt', help='训练好的模型 .pt 文件路径')
    test_image=input("请输入要预测的图片路径：")
    parser.add_argument('-i', '--input', type=str, default=test_image, help='要预测的单张图片路径或图片文件夹路径')
    
    args = parser.parse_args()
    Continue=True
    while Continue:
        main(args)
        ans=input("是否继续预测？(y/n):")
        if ans.lower()!='y':
            exit()
