
from ultralytics import YOLO
import os
import sys

def fix_paths():
    """修复导入路径和文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

def main():
    fix_paths()
    # 选择YOLOv11n模型
    model = YOLO("yolo11n.pt")
    
    # 针对小数据集的优化配置
    train_config = {
        "data": "data.yaml",
        "imgsz": 640,
        "epochs": 80,  # 增加轮数，小模型需要更多时间收敛
        "batch": 16,
        "workers": 2,   # 减少workers，避免小数据集的问题
        "device": 0,
        "project": "runs/train",
        "name": "yolo_exp",
        "exist_ok": True,
        "pretrained": True,
        
        # ⚡ 学习率配置 - 针对小模型调整
        "optimizer": "AdamW",
        "lr0": 0.002,    # 稍高的学习率，小模型收敛快
        "lrf": 0.02,     # 最终学习率
        "cos_lr": True,  # 余弦退火
        
        # 🛡️ 正则化配置
        "weight_decay": 0.001,  # 更强的权重衰减
        "dropout": 0.2,         # 更高的dropout率
        
        # 🔧 数据增强 - 适度增强
        "augment": True,
        "hsv_h": 0.01,
        "hsv_s": 0.6,
        "hsv_v": 0.3,
        "translate": 0.08,
        "scale": 0.4,
        "fliplr": 0.5,
        
        # 📈 训练策略调整
        "patience": 15,         # 增加耐心值， 早停机制
        "save_period": 10,
        "val": True,
        "plots": True,
        
        # 🎯 针对小数据集的特殊配置
        "close_mosaic": 5,      # 更早关闭mosaic增强
        "warmup_epochs": 5,     # 更长的预热
    }
    
    # 开始训练
    results = model.train(**train_config)
    return results
# Windows多进程保护
if __name__ == '__main__':
    main()