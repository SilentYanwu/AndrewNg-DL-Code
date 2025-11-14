import os, sys
import shutil
from ultralytics import YOLO

# ----------------------------------
# 修复路径
# ----------------------------------
def fix_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

fix_paths()

# ----------------------------------
# 配置路径
# ----------------------------------
IMAGE_DIR = "datasets/images/test"
LABEL_DIR = "datasets/labels/test"
VIS_DIR = "datasets/vis/test"
RUNS_DIR = "runs/detect"
TEMP_DIR = "runs/detect/auto_label/labels"
CLASSES_TXT = os.path.join(LABEL_DIR, "classes.txt")

# coco80 类别名称（YOLOv8 内置顺序）
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# 创建目录
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ----------------------------------
# 0. 清空 runs/detect
# ----------------------------------
print("🧹 正在清空 runs/detect ...")

if os.path.exists(RUNS_DIR):
    shutil.rmtree(RUNS_DIR)

print("✔ 已清空 runs/detect")

# ----------------------------------
# 1. YOLO 自动标注（含可视化）
# ----------------------------------
print("🔍 正在使用 YOLOv8 自动标注图片...")
model = YOLO("yolov8s.pt")

results = model.predict(
    source=IMAGE_DIR,
    save=True,          # 保存可视化图像
    save_txt=True,
    save_conf=True,
    project="runs/detect",
    name="auto_label",
    exist_ok=True
)

print("✅ 自动标注完成！")

# ----------------------------------
# 2. 复制 labels
# ----------------------------------
print(f"📂 正在复制标签到 {LABEL_DIR} ...")

if not os.path.exists(TEMP_DIR):
    raise FileNotFoundError("未找到 YOLO 自动生成的标签目录")

count = 0
for file in os.listdir(TEMP_DIR):
    if file.endswith(".txt"):
        src = os.path.join(TEMP_DIR, file)
        dst = os.path.join(LABEL_DIR, file)
        shutil.copy(src, dst)
        count += 1

print(f"✔ 已复制 {count} 个标签文件到 {LABEL_DIR}")

# ----------------------------------
# 3. 保存可视化图像
# ----------------------------------
print("🖼 正在保存可视化检测图片 …")

VIS_SRC = "runs/detect/auto_label"

if os.path.exists(VIS_SRC):
    for file in os.listdir(VIS_SRC):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(os.path.join(VIS_SRC, file), os.path.join(VIS_DIR, file))

print(f"✔ 可视化图片已保存到 {VIS_DIR}")

# ----------------------------------
# 4. 自动生成 classes.txt
# ----------------------------------
print("📝 正在自动生成 LabelImg 专用 classes.txt ...")

with open(CLASSES_TXT, "w", encoding="utf-8") as f:
    for name in COCO_CLASSES:
        f.write(name + "\n")

print(f"🎉 已生成：{CLASSES_TXT}")
print("👍 现在你可以用 LabelImg 打开图片并人工修正标签了！")
