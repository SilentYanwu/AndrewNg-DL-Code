# facenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# 仅供参考学习，实际使用建议用 facenet-pytorch 库
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
        # 目的：基础特征提取，使用小感受野捕获局部细节特征
        # 优势：计算量小，参数少，主要用于通道数的调整和基础特征映射
        # 配置说明：如果structure中指定了1x1分支且输出通道数>0，则创建该分支
        if '1x1' in structure and structure['1x1'] is not None and structure['1x1'][0] > 0:
            self.branch1 = BasicConv2d(in_channels, structure['1x1'][0], kernel_size=1)
        else:
            self.branch1 = None  # 某些Inception变体可能省略此分支以减少参数

        # === 分支 2: 1x1降维 -> 3x3卷积 (瓶颈结构) ===
        # 核心思想：通过"先压缩后扩展"的瓶颈设计大幅减少计算量
        # 计算优势：假设输入256通道，输出256通道
        #   - 直接3x3卷积：256×256×3×3 = 589,824次乘法
        #   - 瓶颈结构(64中间层)：256×64×1×1 + 64×256×3×3 = 163,840 + 147,456 = 311,296次乘法
        #   计算量减少约47%，同时保持相似的表达能力
        self.branch2_1 = BasicConv2d(in_channels, structure['3x3_reduce'][0], kernel_size=1)
        
        # 配置参数：填充和步长，默认使用1像素填充和步长1保持特征图尺寸
        pad_3x3 = structure.get('3x3_pad', 1)  # 默认填充1确保3x3卷积不改变尺寸
        stride_3x3 = structure.get('3x3_stride', 1)  # 默认步长1
        self.branch2_2 = BasicConv2d(structure['3x3_reduce'][0], structure['3x3'][0], 
                                     kernel_size=3, stride=stride_3x3, padding=pad_3x3)

        # === 分支 3: 1x1降维 -> 5x5卷积 (扩展感受野的瓶颈结构) ===
        # 目的：捕获更大范围的上下文信息，适合识别较大尺度的模式
        # 设计原理：同样使用瓶颈结构控制计算复杂度，5x5卷积比3x3有更大感受野但计算成本更高
        self.branch3_1 = BasicConv2d(in_channels, structure['5x5_reduce'][0], kernel_size=1)
        
        pad_5x5 = structure.get('5x5_pad', 2)  # 5x5卷积需要2像素填充来保持尺寸
        stride_5x5 = structure.get('5x5_stride', 1)
        self.branch3_2 = BasicConv2d(structure['5x5_reduce'][0], structure['5x5'][0], 
                                     kernel_size=5, stride=stride_5x5, padding=pad_5x5)

        # === 分支 4: 池化操作 -> 可选的1x1卷积 ===
        # 作用：提供平移不变性，捕获最显著的特征响应
        # 池化类型：最大池化（突出最强特征）或平均池化（平滑特征响应）
        pool_type = structure.get('pool_type', 'max')  # 默认最大池化
        pool_stride = structure.get('pool_stride', 1)  # 池化步长
        
        # 池化层
        if pool_type == 'max':
            self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=1) 
        else:
            self.branch4_pool = nn.AvgPool2d(kernel_size=3, stride=pool_stride, padding=1)

        # 池化后投影层：调整通道数以便与其他分支拼接
        # 如果没有投影层，池化分支将保持输入通道数直接输出
        if 'pool_proj' in structure and structure['pool_proj'] is not None:
            self.branch4_conv = BasicConv2d(in_channels, structure['pool_proj'][0], kernel_size=1)
        else:
            self.branch4_conv = None

    def forward(self, x):
        # 存储四个分支的输出结果，用于后续拼接
        outputs = []
        
        # 分支1处理：直接1x1卷积（如果存在）
        if self.branch1 is not None:
            outputs.append(self.branch1(x))
        
        # 分支2处理：两阶段处理 - 先1x1降维，再3x3特征提取
        branch2_out = self.branch2_1(x)  # 第一阶段：通道压缩
        branch2_out = self.branch2_2(branch2_out)  # 第二阶段：空间特征提取
        outputs.append(branch2_out)

        # 分支3处理：两阶段处理 - 先1x1降维，再5x5大感受野特征提取
        branch3_out = self.branch3_1(x)  # 第一阶段：通道压缩
        branch3_out = self.branch3_2(branch3_out)  # 第二阶段：大范围特征提取
        outputs.append(branch3_out)
        
        # 分支4处理：池化操作 + 可选的通道调整
        branch4_out = self.branch4_pool(x)  # 池化操作
        if self.branch4_conv is not None:
            branch4_out = self.branch4_conv(branch4_out)  # 通道数调整
        outputs.append(branch4_out)
            
        # Inception核心思想：多尺度特征融合
        # 将四个分支在通道维度(dim=1)拼接，形成丰富的多尺度特征表示
        # 输出通道数 = 各分支输出通道数之和
        return torch.cat(outputs, dim=1)
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