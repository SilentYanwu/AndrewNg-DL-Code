from ultralytics import YOLO
import os,sys
# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)

# 在导入本地文件/模型之前调用
fix_paths()

model = YOLO("runs/train/exp_yolo/weights/best.pt")

# 对 test 集推理
model.predict(
    source="datasets/images/test",
    imgsz=640,
    device=0,
    save=True,      # 保存可视化图像
    save_txt=True,  # 保存 YOLO txt（预测标签）
    project="runs/test",
    name="exp_yolo",
    exist_ok=True
)
