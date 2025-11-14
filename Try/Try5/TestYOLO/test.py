from ultralytics import YOLO
import cv2
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

# 1. 加载 YOLOv8 预训练模型（以 yolov8n 为例）
model = YOLO("yolov8n.pt")   # 可改成 yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt

# 2. 读取测试图片
img = "test.jpg"

# 3. 推理
results = model(img)

# 4. 将带框的预测结果保存为 test_out.jpg
annotated_img = results[0].plot()  # 绘制结果（numpy 数组）

cv2.imwrite("test_out.jpg", annotated_img)

print("✅ 预测完成，已保存为 test_out.jpg")
