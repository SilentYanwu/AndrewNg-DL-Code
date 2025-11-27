import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import copy

# =========================================================
# 1. 路径修复
# =========================================================
def fix_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

fix_paths()

# =========================================================
# 2. 全局参数设置
# =========================================================
# 检测设备 (优先使用 GPU )
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")

# 内容图片和风格图片的路径
c_path = "images/louvre.jpg"
s_path = "images/sandstone.jpg"

MAX_DIM = 512                           # 最大边长
OUTPUT_DIR = "output"                   # 输出图片目录
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录

DEFAULT_ITER = 300                      # 默认迭代次数
LR = 0.02                               # 默认学习率
# Pytorch通常 Style 权重需要设得比 Content 高很多
CONTENT_WEIGHT = 1e0                    # Style 权重
STYLE_WEIGHT = 1e6                      # Content 权重
TV_WEIGHT = 1e-6                        # TV 权重

# =========================================================
# 3. 图像加载 / 预处理
# =========================================================
# ImageNet 的均值和方差 (PyTorch VGG 标准)
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

def load_img(path, max_dim=MAX_DIM):
    img = Image.open(path).convert('RGB')
    
    # 计算缩放尺寸
    long_side = max(img.size)
    scale = max_dim / long_side
    new_w = int(img.size[0] * scale)
    new_h = int(img.size[1] * scale)
    
    # 定义 Transform: Resize -> ToTensor (转为 0-1 范围, CHW 格式)
    loader = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor()
    ])
    
    # 增加 Batch 维度: (1, C, H, W)
    image = loader(img).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)      # 去除 batch 维度
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


# 这里的预处理主要是归一化，我们在模型内部处理，
# 所以外部只需要保证是 0-1 的 Tensor 即可。
# 将tensor转为PIL Image对象
def deprocess(tensor):
    image = tensor.to("cpu").clone()  
    image = image.squeeze(0)
    image = image.clamp(0, 1) # 限制在 0-1
    image = transforms.ToPILImage()(image)
    return image # 返回 PIL Image 对象

# =========================================================
# 4. 构建 VGG19 模型与特征提取
# =========================================================
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        # 加载预训练的 VGG19，仅使用 features 部分
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # PyTorch VGG19 layer 索引映射 (根据官方结构)
        # block1_conv1 -> '0', block2_conv1 -> '5', ...
        # block4_conv2 -> '21' (relu4_2)
        # 这里有兴趣可以查看下VGG的架构图
        '''
        索引 (Index)	层类型 (Layer Type)	层的俗称 (Paper Name)	说明
        0	            Conv2d (64 filters)	    block1_conv1	<-- 字典中的 '0' (风格层)
        1	            ReLU		
        2	            Conv2d (64 filters)	    block1_conv2	
        3	            ReLU		
        4	            MaxPool2d		第1个池化层
        5	            Conv2d (128 filters)	block2_conv1	<-- 字典中的 '5' (风格层)
        6	            ReLU		
        7	            Conv2d (128 filters)	block2_conv2	
        8	            ReLU		
        9	            MaxPool2d		第2个池化层
        10	            Conv2d (256 filters)	block3_conv1	<-- 字典中的 '10' (风格层)
        ...	...	...	中间省略 block3_conv2/3/4
        18	            MaxPool2d		第3个池化层
        19	            Conv2d (512 filters)	block4_conv1	<-- 字典中的 '19' (风格层)
        20	            ReLU		
        21	            Conv2d (512 filters)	block4_conv2	<-- 字典中的 '21' (内容层)
        ...	...	...	
        27	            MaxPool2d		第4个池化层
        28	            Conv2d (512 filters)	block5_conv1	<-- 字典中的 '28' (风格层)
        '''
        self.layers = {
            '0': 'block1_conv1',
            '5': 'block2_conv1',
            '10': 'block3_conv1',
            '19': 'block4_conv1',
            '21': 'block4_conv2', # Content Layer
            '28': 'block5_conv1'
        }
        
        # 只保留我们需要用到的层之前的模型部分
        # 模型截断后，后续层的输出会为 None
        self.model = vgg[:29] 
        self.model.to(device)

    def forward(self, x):
        features = {}
        # 手动归一化 (Input is 0-1 RGB)
        mean = normalization_mean.view(-1, 1, 1)
        std = normalization_std.view(-1, 1, 1)
        x = (x - mean) / std
        
        # 逐层前向传播并捕获特定层的输出
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

# 计算 Gram 矩阵
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    # 计算 Gram 矩阵: F * F_transpose
    G = torch.mm(features, features.t())
    # 归一化 (除以元素总数)，这有助于防止大尺寸图片 Loss 爆炸
    return G.div(b * c * h * w)

# =========================================================
# 5. 损失函数
# =========================================================
def compute_loss(model, init_img, style_grams, content_features, 
                 style_weight, content_weight, tv_weight):
    
    current_features = model(init_img)
    
    # 1. Content Loss
    content_loss = 0
    # 我们只用了一层做内容，但为了通用性写成循环
    c_feature = current_features['block4_conv2']
    t_feature = content_features['block4_conv2']
    content_loss += torch.mean((c_feature - t_feature) ** 2)
    
    # 2. Style Loss
    style_loss = 0
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    for layer_name in style_layers:
        current_gram = gram_matrix(current_features[layer_name])
        target_gram = style_grams[layer_name]
        style_loss += torch.mean((current_gram - target_gram) ** 2)
    
    # 3. TV Loss (Total Variation)
    # 计算相邻像素差值
    diff_i = torch.sum(torch.abs(init_img[:, :, :, 1:] - init_img[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(init_img[:, :, 1:, :] - init_img[:, :, :-1, :]))
    tv_loss = (diff_i + diff_j) / (init_img.nelement()) # 归一化

    total_loss = (style_weight * style_loss + 
                  content_weight * content_loss + 
                  tv_weight * tv_loss)
    
    return total_loss, style_loss, content_loss, tv_loss

# =========================================================
# 6. 主训练流程
# =========================================================
def style_transfer(content_path, style_path,
                   iterations=DEFAULT_ITER,
                   style_weight=STYLE_WEIGHT,
                   content_weight=CONTENT_WEIGHT,
                   tv_weight=TV_WEIGHT,
                   lr=LR):

    print("加载图像中...")
    # 若文件不存在请确保路径正确
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print(f"错误: 找不到文件 {content_path} 或 {style_path}")
        return None

    content_img = load_img(content_path)
    style_img = load_img(style_path)
    
    # 显示图片确认
    imshow(content_img, "Content Image")
    imshow(style_img, "Style Image")

    print("提取 VGG19 特征...")
    model = VGGFeatures().eval() # 设为评估模式

    # 计算目标特征 (不计算梯度)
    with torch.no_grad():
        style_features = model(style_img)
        content_features_targets = model(content_img)
        
    # 预计算风格图的 Gram 矩阵
    style_grams = {k: gram_matrix(v) for k, v in style_features.items()}

    # 初始化生成图 (从内容图开始)
    # requires_grad=True 告诉 PyTorch 我们要更新这个 Tensor
    input_img = content_img.clone().requires_grad_(True)
    
    # 使用 Adam 优化器
    optimizer = optim.Adam([input_img], lr=lr)

    print("开始训练...\n")

    for i in range(iterations + 1):
        # 清零梯度
        optimizer.zero_grad()
        
        # 计算 Loss
        total_loss, s_loss, c_loss, tv_loss = compute_loss(
            model, input_img, 
            style_grams, content_features_targets,
            style_weight, content_weight, tv_weight
        )
        
        # 反向传播
        total_loss.backward()
        
        # 更新像素
        optimizer.step()
        
        # 限制像素范围在 [0, 1] 之间 (Projected Gradient Descent)
        with torch.no_grad():
            input_img.clamp_(0, 1)

        if i % 50 == 0:
            print(f"[{i:03d}] total={total_loss.item():.4f} "
                  f"style={s_loss.item() * style_weight:.4f} "
                  f"content={c_loss.item() * content_weight:.4f} "
                  f"tv={tv_loss.item() * tv_weight:.4f}")
            
            save_img = deprocess(input_img.detach())
            save_img.save(f"{OUTPUT_DIR}/step_{i}.jpg")

    final_img = deprocess(input_img.detach())
    final_img.save(f"{OUTPUT_DIR}/final.jpg")
    print(f"\n最终图片已保存到 {OUTPUT_DIR}/final.jpg\n")
    return final_img

# =========================================================
# 7. 运行
# =========================================================
if __name__ == "__main__":

    # 简单检查是否有文件，没有则生成随机噪声图供测试代码运行
    if not os.path.exists(c_path):
        os.makedirs("images", exist_ok=True)
        print("未检测到测试图片，正在生成随机图片用于测试代码...")
        Image.fromarray(np.random.randint(0,255, (300,300,3), dtype='uint8')).save(c_path)
        Image.fromarray(np.random.randint(0,255, (300,300,3), dtype='uint8')).save(s_path)

    final = style_transfer(
        content_path=c_path,
        style_path=s_path,
        iterations=DEFAULT_ITER,
    )
    
    if final:
        plt.imshow(final)
        plt.axis("off")
        plt.show()