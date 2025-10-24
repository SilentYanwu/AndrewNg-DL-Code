import cv2
import numpy as np

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
    # OpenCV读取的图片不会有alpha通道，所以不需要处理RGBA
    
    # 归一化
    if image.max() > 1.0:
        image = image / 255.0
    
    # 展平并转置
    image_flat = image.reshape((1, -1)).T
    
    return image_flat, image