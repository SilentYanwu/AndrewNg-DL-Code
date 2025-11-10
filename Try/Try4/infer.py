# infer.py
import torch
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

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

# 导入本地模块
from resnet_model import create_resnet50 # 导入新的 ResNet 模型

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 关键：必须使用与训练时 *完全相同* 的尺寸和归一化参数
INFER_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_model(model_path, device):
    """
    加载训练好的 ResNet-50 模型
    """
    # 1. 初始化模型结构
    #    重要：use_pretrained=False，因为我们不是加载 ImageNet 权重，
    #    而是加载我们自己训练好的 .pt 文件。
    model = create_resnet50(num_classes=6, use_pretrained=False).to(device)
    
    # 2. 加载训练好的权重
    #    map_location 确保在没有 GPU 的机器上也能加载
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 3. 设置为评估模式
    model.eval()
    return model

def preprocess_image(image_path_or_array):
    """
    图像预处理函数：处理文件路径或numpy数组
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path_or_array}")
    else:
        image = image_path_or_array

    # 1. 颜色空间转换：BGR -> RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    # 2. 转换为PIL Image格式
    image_pil = Image.fromarray(image_rgb)
    
    # 3. 应用与 *验证集/测试集* 完全相同的预处理变换
    preprocess_transform = transforms.Compose([
        transforms.Resize((INFER_SIZE, INFER_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    # 4. 应用变换并增加batch维度 [C, H, W] -> [1, C, H, W]
    image_tensor = preprocess_transform(image_pil).unsqueeze(0)
    
    return image_tensor, image_rgb

def predict_and_show(model, image_path, device, classes):
    """
    执行单张图片预测并可视化结果 (满足您的 'show' 要求)
    """
    try:
        tensor, original_rgb = preprocess_image(image_path)
        tensor = tensor.to(device)
    except ValueError as e:
        print(f"❌ 图像处理错误: {e}")
        return

    # 模型推理
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
        label = str(classes[pred.item()]) # 转换为字符串以便显示
        conf_value = conf.item() * 100

    print(f"📁 文件: {os.path.basename(image_path)}")
    print(f"   🎯 预测类别: {label}")
    print(f"   📊 置信度: {conf_value:.2f}%")

    # 使用 Matplotlib 显示结果 (满足 'show' 要求)
    plt.figure(figsize=(8, 6))
    plt.imshow(original_rgb)
    title_text = f"预测结果: {label} (置信度: {conf_value:.2f}%)"
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def batch_predict(model, input_dir, device, classes):
    """
    批量预测文件夹中的所有图片 (满足您的 '批量识别' 要求)
    """
    print(f"🔍 扫描文件夹: {input_dir}")
    valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_dir, filename)
            
            try:
                tensor, _ = preprocess_image(img_path)
                tensor = tensor.to(device)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                    label = str(classes[pred.item()])
                    conf_value = conf.item() * 100
                    print(f"  📄 {filename:<20} -> 🎯 {label} ({conf_value:.2f}%)")
                    
            except Exception as e:
                print(f"  ❌ 跳过 {filename}: {e}")

def main():
    while True:
    # 交互式输入图片路径
        parser = argparse.ArgumentParser(description="手语 ResNet-50 模型推理工具")
        parser.add_argument('-m', '--model', type=str, default='runs/best_model.pt',
                        help='训练好的模型文件路径 (.pt 文件)')
        test_image = input("📁 请输入要预测的图片路径：")
        parser.add_argument('-i', '--input', type=str, default=test_image,
                            help='要预测的图片路径或图片文件夹路径')
        args = parser.parse_args()
        
        # 1. 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 正在使用设备: {device}")
        
        # 2. 定义类别标签 (0-5)
        classes = [0, 1, 2, 3, 4, 5]
        
        # 3. 加载模型
        if not os.path.exists(args.model):
            print(f"❌ 错误: 找不到模型文件 {args.model}")
            print("请先运行 train_resnet.py 训练模型。")
            return
            
        print(f"📦 正在从 {args.model} 加载模型...")
        try:
            model = load_model(args.model, device)
            print("✅ 模型加载成功!")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return
        
        # 4. 根据输入类型执行不同模式
        if not os.path.exists(args.input):
            print(f"❌ 错误: 输入路径无效: {args.input}")
            return

        if os.path.isfile(args.input):
            print("\n--- 模式: 单张图片预测 ---")
            predict_and_show(model, args.input, device, classes)
            
        elif os.path.isdir(args.input):
            print("\n--- 模式: 批量文件夹预测 ---")
            batch_predict(model, args.input, device, classes)

        # 5. 询问是否继续预测
            cont = input("\n🔄 是否继续预测其他图片？(y/n): ")
            if cont.lower() != 'y':
                print("👋 退出预测程序。")
                break
if __name__ == "__main__":
    main()
    print("👋 谢谢使用")