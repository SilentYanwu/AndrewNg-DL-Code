# model.py
import torch
import torch.nn as nn

class SignCNN(nn.Module):
    def __init__(self, num_classes=6):
        """
        更强大的 CNN 模型定义。
        使用 AdaptiveAvgPool2d 来处理任意输入尺寸。
        
        第L层的输入通道数 = 第L-1层的输出通道数

        第L层的输入通道数 = 第L层单个滤波器的层数

        第L层的滤波器个数 = 第L层的输出通道数
        
        不过，我们在代码中，主要保证上一层的输出通道数 = 下一层的输入通道数即可。
        参数:
            num_classes: 分类类别数量，默认为6类
        """
        # 调用父类构造函数
        super(SignCNN, self).__init__()
        
        # 第一个卷积块：提取基础特征
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # 输入通道3(RGB)，输出16通道，5x5卷积核
            nn.BatchNorm2d(16),  # 批量归一化，加速训练并提高稳定性
            nn.ReLU(),           # ReLU激活函数，引入非线性
            nn.MaxPool2d(2, 2)   # 2x2最大池化，特征图尺寸减半
        )
        
        # 第二个卷积块：提取更复杂的特征
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输入16通道，输出32通道，3x3卷积核
            nn.BatchNorm2d(32),  # 批量归一化
            nn.ReLU(),           # ReLU激活函数
            nn.MaxPool2d(2, 2)   # 2x2最大池化，特征图尺寸再次减半
        )
        
        # 第三个卷积块：提取高级语义特征
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输入32通道，输出64通道，3x3卷积核
            nn.BatchNorm2d(64),  # 批量归一化
            nn.ReLU(),           # ReLU激活函数
            nn.MaxPool2d(2, 2)   # 2x2最大池化，特征图尺寸第三次减半
        )
        ## 新增部分：
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )
                
        # ⭐️ 核心改进：自适应平均池化
        # 无论输入图像尺寸如何，经过3次MaxPool后，
        # 自适应池化都会将特征图统一转换为8x8大小
        # 这使得模型能够处理任意尺寸的输入图像
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Dropout层：防止过拟合，随机丢弃50%的神经元
        self.dropout = nn.Dropout(0.5)
        
        # 全连接分类器
        # 输入维度：128通道 * 8高度 * 8宽度 = 8192
        # 输出维度：num_classes个类别
        self.classifier = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        """
        前向传播过程
        
        参数:
            x: 输入张量，形状为 [batch_size, 3, height, width]
            
        返回:
            输出张量，形状为 [batch_size, num_classes]
        """
        # 通过三个卷积块提取特征
        x = self.conv_block1(x)  # 第一次特征提取和下采样
        x = self.conv_block2(x)  # 第二次特征提取和下采样
        x = self.conv_block3(x)  # 第三次特征提取和下采样
        ## 新增部分：
        x = self.conv_block4(x)
        # ⭐️ 应用自适应池化，统一特征图尺寸为8x8
        x = self.adaptive_pool(x)
        
        # 应用Dropout防止过拟合
        x = self.dropout(x)
        
        # 将特征图展平为一维向量，供全连接层使用
        # x.reshape(x.size(0), -1) 保持batch_size不变，将所有特征展平
        x = x.reshape(x.size(0), -1)
        
        # 通过分类器得到最终的类别预测
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    """
    模型测试代码
    验证模型是否能正确处理不同尺寸的输入
    """
    # 创建模型实例
    model = SignCNN()
    
    # 创建测试数据：模拟64x64和128x128两种尺寸的输入图像
    test_img_64 = torch.randn(1, 3, 64, 64)    # 批量大小1，3通道，64x64像素
    test_img_128 = torch.randn(1, 3, 128, 128) # 批量大小1，3通道，128x128像素
    
    # 测试模型对不同尺寸输入的处理能力
    print(f"输入 64x64, 输出: {model(test_img_64).shape}")   # 预期输出: [1, 6]
    print(f"输入 128x128, 输出: {model(test_img_128).shape}") # 预期输出: [1, 6]
    print("✅ 模型已支持任意尺寸输入。")