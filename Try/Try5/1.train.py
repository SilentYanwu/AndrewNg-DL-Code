from ultralytics import YOLO
import os, sys

# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

def main():
    """主要的训练函数"""
    fix_paths()
    
    # 选择模型
    model = YOLO("yolov8s.pt")
    
    # 开始训练
    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        workers=4,  # 如果问题持续，可以尝试减少workers数量
        device=0,
        project="runs/train",
        name="exp_yolo",
        exist_ok=True,
        # 模型配置
        pretrained=True,       # 使用预训练权重
        # 优化器设置
        optimizer="auto",    # 优化器选择(auto, SGD, Adam, AdamW等)
        lr0=0.01,           # 初始学习率
        lrf=0.01,           # 最终学习率系数(lr0 * lrf)
        
        # 数据增强
        augment=True,       # 是否启用数据增强
        hsv_h=0.015,        # 色调增强幅度
        hsv_s=0.7,          # 饱和度增强幅度  
        hsv_v=0.4,          # 明度增强幅度
        translate=0.1,      # 平移增强幅度
        scale=0.5,          # 缩放增强幅度 
)
print("✅ 训练完成！")

# Windows多进程保护
if __name__ == '__main__':
    # 在Windows上使用多进程时必须要有这个保护
    main()
