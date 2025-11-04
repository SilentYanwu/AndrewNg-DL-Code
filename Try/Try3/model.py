# model.py
import torch
import torch.nn as nn

class SignCNN(nn.Module):
    def __init__(self, num_classes=6):
        """
        更强大的 CNN 模型定义。
        使用 AdaptiveAvgPool2d 来处理任意输入尺寸。
        """
        super(SignCNN, self).__init__()
        
        # 卷积块1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 尺寸 /2
        )
        
        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 尺寸 /2
        )
        
        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 尺寸 /2
        )
        
        # ⭐️ 核心改进：自适应平均池化
        # 无论输入图像多大，经过3次MaxPool后，
        # 这里的 adaptive_pool 都会将其转换为 8x8 的特征图
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.dropout = nn.Dropout(0.5)
        
        # 我们的分类器现在总是接收 64 * 8 * 8 的输入
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = self.adaptive_pool(x)  # ⭐️ 应用自适应池化
        
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1) # 展平
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # 测试模型是否能处理不同尺寸
    model = SignCNN()
    test_img_64 = torch.randn(1, 3, 64, 64)
    test_img_128 = torch.randn(1, 3, 128, 128)
    
    print(f"输入 64x64, 输出: {model(test_img_64).shape}")   # [1, 6]
    print(f"输入 128x128, 输出: {model(test_img_128).shape}") # [1, 6]
    print("✅ 模型已支持任意尺寸输入。")