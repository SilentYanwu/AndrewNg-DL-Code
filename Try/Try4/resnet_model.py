# resnet_model.py
import torch
import torch.nn as nn
from torchvision import models

def create_resnet50(num_classes=6, use_pretrained=True, freeze_layers=True):
    """
    创建并配置一个 ResNet-50 模型。

    参数:
        num_classes (int): 输出类别的数量 (我们的手势是 0-5, 所以是 6)。
        use_pretrained (bool): 是否加载在 ImageNet 上预训练的权重。
        freeze_layers (bool): 是否冻结预训练的卷积层 (只训练分类头)。
                               这在 fine-tuning 时很常见。
    
    返回:
        model: 配置好的 ResNet-50 模型
    """
    
    # 1. 加载 ResNet-50 模型
    if use_pretrained:
        # 使用推荐的现代方法加载预训练权重
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        # 加载没有预训练权重的模型
        model = models.resnet50(weights=None)

    # 2. (可选) 冻结所有卷积层
    # 当数据集较小且与 ImageNet 差异较大时，通常先只训练分类头
    if freeze_layers and use_pretrained:
        for param in model.parameters():
            param.requires_grad = False
            
    # 3. 替换最后的全连接层 (分类头)
    # ResNet-50 的最后分类层叫做 'fc'
    # 它的输入特征数是 2048 (model.fc.in_features)
    
    # 获取 'fc' 层的输入特征数
    num_ftrs = model.fc.in_features
    
    # 将 'fc' 层替换为一个新的线性层
    # 新的层默认 requires_grad = True
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"✅ ResNet-50 模型创建成功。")
    print(f"   - 预训练权重: {use_pretrained}")
    print(f"   - 冻结卷积层: {freeze_layers}")
    print(f"   - 输出类别: {num_classes}")
    
    return model

if __name__ == "__main__":
    """
    测试模型是否能正确构建
    """
    # 测试创建预训练模型
    model_pre = create_resnet50(num_classes=6, use_pretrained=True)
    
    # 测试创建非预训练模型 (用于推理加载)
    model_scratch = create_resnet50(num_classes=6, use_pretrained=False)
    
    # 测试模型前向传播
    # 我们的数据是 64x64，ResNet 的自适应池化层会处理这个尺寸
    test_tensor = torch.randn(1, 3, 64, 64)
    output = model_pre(test_tensor)
    
    print(f"\n测试输入 [1, 3, 64, 64], 输出形状: {output.shape}") # 应为 [1, 6]
    assert output.shape == (1, 6)
    print("✅ 模型构建和前向传播测试通过。")