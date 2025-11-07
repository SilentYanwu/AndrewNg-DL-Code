from torchvision import transforms

IMG_SIZE = 224

train_transform = transforms.Compose([
    # =========================
    # 几何增强部分
    # =========================
    # 随机旋转 ±30°
    transforms.RandomRotation(30),

    # 随机透视变换（模拟不同拍摄角度）
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),

    # 随机水平翻转 + 垂直翻转（镜像）
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),

    # 随机裁剪与缩放（关注主体区域）
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),

    # =========================
    # 颜色增强部分
    # =========================
    transforms.ColorJitter(
        brightness=0.4,  # 亮度变化范围
        contrast=0.4,    # 对比度变化
        saturation=0.3,  # 饱和度变化
        hue=0.05         # 色相微调
    ),

    # =========================
    # 模糊增强
    # =========================
    # 随机高斯模糊（模拟摄像头虚焦）
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),

    # =========================
    # 转 Tensor + 标准化
    # =========================
    transforms.ToTensor(),

    # 随机遮挡（Cutout/RandomErasing）
    transforms.RandomErasing(
        p=0.5,               # 50% 概率遮挡
        scale=(0.02, 0.2),   # 遮挡区域相对面积范围
        ratio=(0.3, 3.3),    # 遮挡矩形宽高比
        value='random'       # 遮挡区域随机颜色
    ),

    # 归一化（适配常见模型）
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 验证集和测试集不需要增强，只需基础预处理
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),    # 调整图像尺寸
    transforms.ToTensor(),                       # 将PIL图像转为Tensor
    transforms.Normalize(                       # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    
])