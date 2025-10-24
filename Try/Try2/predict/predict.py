'''
PyTorch GPU 版本 - 交互式模型预测脚本
支持选择不同格式的模型文件
'''
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

# 在导入本地之前调用
fix_paths()

# 设置 Matplotlib 使用支持中文的字体（Windows 推荐 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# =========================================================
# 一、图片预处理函数
# =========================================================
def preprocess_image(image_path, target_size=(64, 64)):
    """
    通用图片预处理函数，支持多种格式
    """
    # 使用OpenCV读取图片
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换颜色空间 BGR -> RGB (OpenCV默认是BGR，但我们需要RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用OpenCV调整尺寸到64x64
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # 处理不同通道数
    if len(image.shape) == 2:  # 灰度图
        image = np.stack([image, image, image], axis=-1)
    
    # 归一化
    if image.max() > 1.0:
        image = image / 255.0
    
    # 展平并转置 (保持与训练时相同的格式)
    image_flat = image.reshape((1, -1)).T
    
    return image_flat, image

# =========================================================
# 二、定义相同的神经网络模型
# =========================================================
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size=12288, hidden1_size=25, hidden2_size=12, output_size=6):
        super(ThreeLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# =========================================================
# 三、模型加载函数
# =========================================================
def load_model_by_type(model_type="auto"):
    """
    根据类型加载模型
    model_type: "dict" - 字典版, "full" - 完整版, "auto" - 自动选择
    """
    model_files = {
        "dict": "three_layer_nn_model.pth",
        "full": "three_layer_nn_full_model.pth"
    }
    
    if model_type == "auto":
        # 自动选择：优先使用完整版，如果没有则使用字典版
        if os.path.exists(model_files["full"]):
            model_path = model_files["full"]
            print("✅ 自动选择: 完整版模型")
        elif os.path.exists(model_files["dict"]):
            model_path = model_files["dict"]
            print("✅ 自动选择: 字典版模型")
        else:
            print("❌ 没有找到任何模型文件")
            return None
    else:
        model_path = model_files.get(model_type)
        if not model_path or not os.path.exists(model_path):
            print(f"❌ 找不到指定的模型文件: {model_files.get(model_type)}")
            return None
    
    print(f"📁 加载模型: {model_path}")
    return load_model(model_path)

def load_model(model_path):
    """
    加载指定路径的模型
    """
    try:
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location=device)
        
        # 判断模型格式并加载
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("✅ 检测到完整版模型格式")
            
            # 从checkpoint中获取模型配置
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                model = ThreeLayerNN(
                    input_size=config.get('input_size', 12288),
                    hidden1_size=config.get('hidden1_size', 25),
                    hidden2_size=config.get('hidden2_size', 12),
                    output_size=config.get('output_size', 6)
                ).to(device)
            else:
                model = ThreeLayerNN().to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 显示准确率信息
            if 'test_accuracy' in checkpoint:
                print(f"📊 模型测试准确率: {checkpoint['test_accuracy']:.2f}%")
                
        else:
            print("✅ 检测到字典版模型格式")
            model = ThreeLayerNN().to(device)
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ 模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

# =========================================================
# 四、预测函数
# =========================================================
def predict_image(model, image_path):
    """预测单张图片"""
    # 预处理图片
    image_flat, original_image = preprocess_image(image_path)
    
    # 转换为PyTorch张量
    image_tensor = torch.FloatTensor(image_flat.T).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        prediction = predicted.cpu().numpy()[0]
        confidence_score = confidence.cpu().numpy()[0]
    
    return prediction, confidence_score, original_image

# =========================================================
# 五、检查模型文件
# =========================================================
def check_model_files():
    """检查可用的模型文件"""
    model_files = {
        "dict": "three_layer_nn_model.pth",
        "full": "three_layer_nn_full_model.pth"
    }
    
    available_models = []
    for model_type, filename in model_files.items():
        if os.path.exists(filename):
            available_models.append(model_type)
            print(f"✅ {model_type}版: {filename} (存在)")
        else:
            print(f"❌ {model_type}版: {filename} (不存在)")
    
    return available_models

# =========================================================
# 六、单张图片预测
# =========================================================
def single_image_prediction(model):
    """单张图片预测"""
    print("\n📸 单张图片预测")
    print("-" * 30)
    
    # 获取图片路径
    default_path = "images/5.png"
    if os.path.exists(default_path):
        image_path = input(f"请输入图片路径 [默认: {default_path}]: ").strip()
        if not image_path:
            image_path = default_path
    else:
        image_path = input("请输入图片路径: ").strip()
    
    # 移除可能的引号
    image_path = image_path.strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print(f"❌ 图片路径不存在: {image_path}")
        return
    
    try:
        prediction, confidence, original_image = predict_image(model, image_path)
        
        # 显示结果
        plt.figure(figsize=(10, 6))
        plt.imshow(original_image)
        plt.title(f"预测结果: {prediction} (置信度: {confidence:.2%})", fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"🎯 预测结果: {prediction}")
        print(f"📊 置信度: {confidence:.2%}")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")

# =========================================================
# 七、批量图片预测
# =========================================================
def batch_image_prediction(model):
    """批量图片预测"""
    print("\n📁 批量图片预测")
    print("-" * 30)
    
    folder_path = input("请输入图片文件夹路径: ").strip().strip('"').strip("'")
    
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹路径不存在: {folder_path}")
        return
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"❌ 在文件夹中没有找到图片文件")
        return
    
    print(f"🔍 找到 {len(image_files)} 张图片")
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for i, image_file in enumerate(image_files, 1):
        try:
            prediction, confidence, _ = predict_image(model, image_file)
            results.append((image_file, prediction, confidence))
            
            filename = os.path.basename(image_file)
            print(f"{i:2d}/{len(image_files)}: {filename:20s} → 预测: {prediction}, 置信度: {confidence:.2%}")
            
            # 如果文件名包含真实标签（例如: "5_cat.png"），可以进行比较
            # 这里只是示例，实际使用时需要根据您的文件名格式调整
            if '_' in filename:
                true_label = filename.split('_')[0]
                if true_label.isdigit() and int(true_label) == prediction:
                    correct_predictions += 1
                total_predictions += 1
                
        except Exception as e:
            print(f"❌ 处理 {os.path.basename(image_file)} 时出错: {e}")
            results.append((image_file, None, 0.0))
    
    # 如果有真实标签比较，显示准确率
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n📈 批量预测准确率: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    return results

# =========================================================
# 八、模型信息显示
# =========================================================
def show_model_info(model):
    """显示模型信息"""
    print("\n📋 模型信息")
    print("-" * 30)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  模型架构: ThreeLayerNN")
    print(f"🔢 总参数量: {total_params:,}")
    print(f"🎯 可训练参数: {trainable_params:,}")
    print(f"⚙️  设备: {device}")
    
    # 显示各层信息
    print("\n📊 各层信息:")
    for name, layer in model.named_children():
        if hasattr(layer, 'weight'):
            print(f"  {name}: {tuple(layer.weight.shape)}")

# =========================================================
# 九、主交互界面
# =========================================================
def main_interactive():
    """主交互界面"""
    current_model = None
    
    while True:
        print("\n" + "="*60)
        print("🎯 PyTorch 三层神经网络 - 交互式预测系统")
        print("="*60)
        
        # 显示当前加载的模型
        if current_model:
            print(f"✅ 当前已加载模型")
        else:
            print("❌ 当前未加载模型")
        
        print("\n请选择操作:")
        print("1. 📁 选择并加载模型")
        print("2. 📸 单张图片预测")
        print("3. 📁 批量图片预测")
        print("4. 📋 显示模型信息")
        print("5. 🔍 检查模型文件")
        print("6. 🚪 退出系统")
        
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == '1':
            # 选择并加载模型
            print("\n📁 选择模型类型:")
            print("1. 自动选择 (推荐)")
            print("2. 字典版模型 (three_layer_nn_model.pth)")
            print("3. 完整版模型 (three_layer_nn_full_model.pth)")
            
            model_choice = input("请选择模型类型 (1-3): ").strip()
            
            if model_choice == '1':
                current_model = load_model_by_type("auto")
            elif model_choice == '2':
                current_model = load_model_by_type("dict")
            elif model_choice == '3':
                current_model = load_model_by_type("full")
            else:
                print("❌ 无效选择")
            
        elif choice == '2':
            # 单张图片预测
            if current_model:
                single_image_prediction(current_model)
            else:
                print("❌ 请先加载模型")
                
        elif choice == '3':
            # 批量图片预测
            if current_model:
                batch_image_prediction(current_model)
            else:
                print("❌ 请先加载模型")
                
        elif choice == '4':
            # 显示模型信息
            if current_model:
                show_model_info(current_model)
            else:
                print("❌ 请先加载模型")
                
        elif choice == '5':
            # 检查模型文件
            available_models = check_model_files()
            if not available_models:
                print("❌ 没有找到可用的模型文件")
            else:
                print(f"✅ 可用的模型: {', '.join(available_models)}")
                
        elif choice == '6':
            # 退出系统
            print("👋 感谢使用，再见！")
            break
            
        else:
            print("❌ 无效选择，请重新输入")

# =========================================================
# 十、主程序入口
# =========================================================
if __name__ == "__main__":
    print("🚀 PyTorch 三层神经网络预测系统")
    print("💡 支持字典版和完整版模型")
    
    # 自动检查模型文件
    available_models = check_model_files()
    
    if available_models:
        print(f"\n✅ 发现 {len(available_models)} 个可用模型")
        # 启动交互界面
        main_interactive()
    else:
        print("\n❌ 没有找到任何模型文件，请先运行训练脚本")
        print("💡 请确保以下文件存在:")
        print("   - three_layer_nn_model.pth (字典版)")
        print("   - three_layer_nn_full_model.pth (完整版)")