import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 基础组件：BasicConv2d
# 作用：这是深度学习中最标准的 "卷积-批归一化-激活" 三明治结构。
# ---------------------------------------------------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        # 1. 卷积层：提取特征
        # bias=False 是因为后面紧跟了 BN 层，BN 层有通过 bias，所以卷积层不需要
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # 2. Batch Normalization：加速收敛，防止梯度消失，让训练更稳定
        self.bn = nn.BatchNorm2d(out_channels, eps=0.00001, momentum=0.1)
        # 3. ReLU：引入非线性，让神经网络能拟合复杂函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ---------------------------------------------------------------------
# 核心组件：InceptionBlock
# 来源：GoogLeNet (Inception v1)
# 哲学：不要人为决定是用 3x3 卷积还是 5x5 卷积，而是全部都用，
#       让网络自己去学习哪种特征最好，最后把结果拼起来。
# ---------------------------------------------------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, structure):
        super(InceptionBlock, self).__init__()
        
        # === 分支 1: 1x1 卷积 ===
        # 作用：直接提取特征，保留空间信息，或者用于改变通道数。
        # 有些层（如 inc_1c）为了减少参数量，可能会去掉这个分支。
        if '1x1' in structure and structure['1x1'] is not None and structure['1x1'][0] > 0:
            self.branch1 = BasicConv2d(in_channels, structure['1x1'][0], kernel_size=1)
        else:
            self.branch1 = None

        # === 分支 2: 1x1 -> 3x3 (瓶颈层结构) ===
        # 关键点：为什么不直接用 3x3？
        # 答：先用 1x1 卷积把通道数(channels) 降下来（3x3_reduce），
        #     然后再做昂贵的 3x3 卷积。这大大减少了计算量！这叫 "Bottleneck" 层。
        self.branch2_1 = BasicConv2d(in_channels, structure['3x3_reduce'][0], kernel_size=1)
        
        # 读取配置中的 padding 和 stride
        pad_3x3 = structure.get('3x3_pad', 1) 
        stride_3x3 = structure.get('3x3_stride', 1)
        self.branch2_2 = BasicConv2d(structure['3x3_reduce'][0], structure['3x3'][0], 
                                     kernel_size=3, stride=stride_3x3, padding=pad_3x3)

        # === 分支 3: 1x1 -> 5x5 (瓶颈层结构) ===
        # 作用：提取更大感受野的特征。同样先用 1x1 降维。
        self.branch3_1 = BasicConv2d(in_channels, structure['5x5_reduce'][0], kernel_size=1)
        
        pad_5x5 = structure.get('5x5_pad', 2)
        stride_5x5 = structure.get('5x5_stride', 1)
        self.branch3_2 = BasicConv2d(structure['5x5_reduce'][0], structure['5x5'][0], 
                                     kernel_size=5, stride=stride_5x5, padding=pad_5x5)

        # === 分支 4: 池化 -> (可选 1x1) ===
        # 作用：传统的池化层也是提取特征的好帮手。
        # 为了能和其他分支拼接，池化后通常接一个 1x1 卷积来调整通道数。
        pool_type = structure.get('pool_type', 'max')
        pool_stride = structure.get('pool_stride', 1)
        
        if pool_type == 'max':
            self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=1) 
        else:
            self.branch4_pool = nn.AvgPool2d(kernel_size=3, stride=pool_stride, padding=1)

        # 池化后的投影层 (1x1 conv)
        if 'pool_proj' in structure and structure['pool_proj'] is not None:
            self.branch4_conv = BasicConv2d(in_channels, structure['pool_proj'][0], kernel_size=1)
        else:
            self.branch4_conv = None

    def forward(self, x):
        outputs = []
        # 1. 计算分支 1
        if self.branch1 is not None:
            outputs.append(self.branch1(x))
        
        # 2. 计算分支 2 (1x1 -> 3x3)
        outputs.append(self.branch2_2(self.branch2_1(x)))

        # 3. 计算分支 3 (1x1 -> 5x5)
        outputs.append(self.branch3_2(self.branch3_1(x)))
        
        # 4. 计算分支 4 (Pool -> 1x1)
        out4 = self.branch4_pool(x)
        if self.branch4_conv is not None:
            out4 = self.branch4_conv(out4)
        outputs.append(out4)
            
        # 5. 拼接 (Concatenate)
        # 这里的 dim=1 指的是通道维度 (Batch, Channel, Height, Width)
        # Inception 的精髓：将不同尺度的特征在深度方向上堆叠起来
        return torch.cat(outputs, 1)

# ---------------------------------------------------------------------
# 主模型：FaceNet
# ---------------------------------------------------------------------
class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        
        # === 1. Stem (主干部分) ===
        # 在进入昂贵的 Inception 块之前，先用普通的卷积层快速缩小图片尺寸
        # 输入: (3, 96, 96)
        self.pad1 = nn.ZeroPad2d(3) # 只有为了匹配特定权重才需要这么大的 padding
        
        # Conv1: 大卷积核 (7x7) + 步长 2 -> 快速降采样
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=0) 
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2: 1x1 卷积在这里的作用是增加非线性并在增加通道前整理特征
        self.conv2_1 = BasicConv2d(64, 64, kernel_size=1)
        self.conv2_2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 到这里，图片已经变小了很多，特征通道变多了 (192 channels)

        # === 2. Inception Blocks (特征提取核心) ===
        
        # Block 1a
        # 输入 192 -> 输出 256 (64+128+32+32)
        self.inc_1a = InceptionBlock(192, {
            '1x1': [64], 
            '3x3_reduce': [96], '3x3': [128],  # 先压到96，再卷出128
            '5x5_reduce': [16], '5x5': [32],   # 先压到16，再卷出32
            'pool_proj': [32]
        }) 

        # Block 1b
        # 输入 256 -> 输出 320 (64+128+64+64)
        self.inc_1b = InceptionBlock(256, {
            '1x1': [64], 
            '3x3_reduce': [96], '3x3': [128], 
            '5x5_reduce': [32], '5x5': [64], 
            'pool_proj': [64],
            'pool_type': 'avg', 'pool_stride': 1 # 注意：这里 Stride=1 保持尺寸不变
        }) 

        # Block 1c
        # 输入 320 -> 输出 640
        # 这个块比较特殊， stride=2，意味着它不仅提取特征，还负责把图片尺寸缩小一半
        # 通道数计算: 0(1x1没有) + 256(3x3) + 64(5x5) + 320(Pool直接透传) = 640
        self.inc_1c = InceptionBlock(320, {
             '1x1': None, # 这一层没有 1x1 分支
             '3x3_reduce': [128], '3x3': [256], '3x3_stride': 2, # 下采样
             '5x5_reduce': [32], '5x5': [64], '5x5_stride': 2,   # 下采样
             'pool_type': 'max', 'pool_stride': 2,               # 下采样
             'pool_proj': None # 池化后不接卷积，直接拼接，导致通道数剧增
        })

        # === 3. 全连接层 (Embedding Generation) ===
        # 把卷积得到的特征图 (Feature Map) 展平成一维向量
        # 输入维度计算：640 (channels) * 6 (height) * 6 (width) = 23040
        # 输出维度：128 (FaceNet 的标准 Embedding 长度)
        self.last_linear = nn.Linear(640 * 6 * 6, 128)
        
    def forward(self, x):
        # 前向传播流程
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.inc_1a(x)
        x = self.inc_1b(x)
        x = self.inc_1c(x)
        
        # Flatten: 将 (Batch, 640, 6, 6) 变成 (Batch, 23040)
        x = x.view(x.size(0), -1)
        
        # 生成 128维 向量
        x = self.last_linear(x) 
        
        # === 4. L2 Normalization (关键步骤) ===
        # 将向量投影到半径为 1 的超球面上。
        # 这一步至关重要！因为 Triplet Loss 依靠欧氏距离，
        # 只有归一化后，计算出的距离才代表"相似度"，而不是"向量长度"。
        x = F.normalize(x, p=2, dim=1)
        
        return x

def load_model_wrapper():
    model = FaceNet()
    return model