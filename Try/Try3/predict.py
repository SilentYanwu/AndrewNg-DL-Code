# predict.py
import torch
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

# 从 model.py 导入共享的CNN模型
from model import SignCNN

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 使用模型训练时的尺寸进行推理（保持一致性）
INFER_SIZE = 64 

def fix_paths():
    """
    修复导入路径和文件路径，确保模块正确导入
    
    功能:
        - 将当前目录添加到Python路径
        - 切换到当前工作目录
    """
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 将当前目录添加到Python路径，确保能正确导入本地模块
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)

def load_model(model_path, device):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型文件路径
        device: 运行设备 (CPU/GPU)
        
    返回:
        model: 加载好权重的模型（设置为评估模式）
    """
    # 初始化模型结构（必须与训练时一致）
    model = SignCNN(num_classes=6).to(device)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 设置为评估模式（关闭dropout等训练专用层）
    model.eval()
    
    return model

def preprocess_image(image_path_or_array):
    """
    图像预处理函数：处理文件路径或numpy数组
    
    参数:
        image_path_or_array: 图片文件路径或numpy数组
        
    返回:
        image_tensor: 预处理后的张量 [1, C, H, W]
        image_rgb: 原始RGB图像（用于显示）
        
    异常:
        ValueError: 当无法读取图片文件时抛出
    """
    # 处理不同类型的输入
    if isinstance(image_path_or_array, str):
        # 从文件路径读取图片
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path_or_array}")
    else:
        # 假设输入已经是numpy数组（BGR格式）
        image = image_path_or_array

    # 1. 颜色空间转换：BGR -> RGB
    if len(image.shape) == 2: 
        # 灰度图转RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # BGR图转RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # 2. 转换为PIL Image格式（便于使用torchvision变换）
    image_pil = Image.fromarray(image_rgb)
    
    # 3. 应用与验证集/测试集完全相同的预处理变换
    preprocess_transform = transforms.Compose([
        transforms.Resize((INFER_SIZE, INFER_SIZE)),  # 调整到模型训练时的尺寸
        transforms.ToTensor(),                        # 转换为Tensor并归一化到[0,1]
        transforms.Normalize(                       # 归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
    ])
    
    # 4. 应用变换并增加batch维度 [C, H, W] -> [1, C, H, W]
    image_tensor = preprocess_transform(image_pil).unsqueeze(0)
    
    return image_tensor, image_rgb

def predict_and_show(model, image_path, device, classes):
    """
    执行单张图片预测并可视化结果
    
    参数:
        model: 训练好的模型
        image_path: 图片文件路径
        device: 运行设备
        classes: 类别标签列表
    """
    try:
        # 预处理图像
        tensor, original_rgb = preprocess_image(image_path)
        tensor = tensor.to(device)
    except ValueError as e:
        print(f"❌ 图像处理错误: {e}")
        return

    # 模型推理（不计算梯度以提高效率）
    with torch.no_grad():
        # 前向传播
        outputs = model(tensor)
        
        # 计算softmax概率
        probs = torch.softmax(outputs, dim=1)
        
        # 获取最大概率值和对应的类别索引
        conf, pred = torch.max(probs, 1)
        
        # 获取预测标签和置信度
        label = classes[pred.item()]
        conf_value = conf.item() * 100  # 转换为百分比

    # 打印预测结果
    print(f"📁 文件: {os.path.basename(image_path)}")
    print(f"  🎯 预测类别: {label}")
    print(f"  📊 置信度: {conf_value:.2f}%")

    # 使用Matplotlib显示结果（比OpenCV窗口更友好）
    plt.figure(figsize=(8, 6))
    plt.imshow(original_rgb)  # 显示原始RGB图像
    
    # 设置标题
    title_text = f"预测结果: {label} (置信度: {conf_value:.2f}%)"
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.axis("off")  # 隐藏坐标轴
    
    # 显示图像
    plt.tight_layout()
    plt.show()

def batch_predict(model, input_dir, device, classes):
    """
    批量预测文件夹中的所有图片
    
    参数:
        model: 训练好的模型
        input_dir: 输入文件夹路径
        device: 运行设备
        classes: 类别标签列表
    """
    print(f"🔍 扫描文件夹: {input_dir}")
    
    # 支持的图片格式
    valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    
    for filename in os.listdir(input_dir):
        # 检查文件扩展名
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_dir, filename)
            
            try:
                # 预处理和预测
                tensor, _ = preprocess_image(img_path)
                tensor = tensor.to(device)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                    # 打印结果（包含置信度）
                    label = classes[pred.item()]
                    conf_value = conf.item() * 100
                    print(f"📄 {filename} -> 🎯 {label} ({conf_value:.2f}%)")
                    
            except Exception as e:
                print(f"❌ 跳过 {filename}: {e}")

def main(args):
    """
    主预测函数
    
    功能:
        - 加载模型
        - 根据输入类型（文件/文件夹）执行预测
        - 处理预测结果
    """
    # 设置运行设备（优先GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    # 定义类别标签（应与训练时一致）
    classes = [0, 1, 2, 3, 4, 5]  # 可以根据实际类别名称修改
    
    # 加载模型
    print(f"📦 正在从 {args.model} 加载模型...")
    try:
        model = load_model(args.model, device)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 根据输入类型执行不同的预测模式
    if os.path.isfile(args.input):
        print("--- 单张图片预测模式 ---")
        predict_and_show(model, args.input, device, classes)
        
    elif os.path.isdir(args.input):
        print("--- 批量文件夹预测模式 ---")
        batch_predict(model, args.input, device, classes)
    else:
        print(f"❌ 错误: 输入路径无效: {args.input}")

if __name__ == "__main__":
    # 修复路径（在导入本地模块之前）
    fix_paths()
        
    # 交互式循环预测
    continue_predicting = True
    while continue_predicting:
    
        # 设置命令行参数
        parser = argparse.ArgumentParser(description="手语CNN模型预测工具")
        parser.add_argument('-m', '--model', type=str, default='runs/best_model.pt',
                        help='训练好的模型文件路径 (.pt 文件)')
        
        # 交互式输入图片路径
        test_image = input("📁 请输入要预测的图片路径：")
        parser.add_argument('-i', '--input', type=str, default=test_image,
                        help='要预测的图片路径或图片文件夹路径')
        
        # 解析参数
        args = parser.parse_args()

        # 执行预测
        main(args)
        
        # 询问用户是否继续
        answer = input("\n🔄 是否继续预测？(y/n): ").lower()
        if answer != 'y':
            continue_predicting = False
            print("👋 程序退出，感谢使用！")