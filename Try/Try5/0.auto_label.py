# -*- coding: utf-8 -*-
"""
自动使用 YOLOv8 对图片进行自动标注
支持 train/val/test 分类标注并自动生成 LabelImg 的 classes.txt
"""
import os
import sys
import shutil
from ultralytics import YOLO

# ----------------------------------
# 修复当前脚本路径
# ----------------------------------
def fix_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

fix_paths()

# ----------------------------------
# 配置
# ----------------------------------
BASE_IMAGE_DIR = "datasets/images"
BASE_LABEL_DIR = "datasets/labels"
BASE_VIS_DIR   = "datasets/vis"
RUNS_DIR = "runs/detect"
YOLO_TEMP_LABEL_DIR = "runs/detect/auto_label/labels"

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

# ----------------------------------
# 清空 runs/detect
# ----------------------------------
def clear_runs():
    if os.path.exists(RUNS_DIR):
        print("🧹 正在清空 runs/detect ...")
        shutil.rmtree(RUNS_DIR)
    print("✔ 已清空 runs/detect\n")


# ----------------------------------
# 自动生成 classes.txt（仅生成一次）
# ----------------------------------
def generate_classes_txt(label_dir):
    classes_path = os.path.join(label_dir, "classes.txt")
    if not os.path.exists(classes_path):
        print("📝 正在生成 classes.txt ...")
        with open(classes_path, "w", encoding="utf-8") as f:
            for name in COCO_CLASSES:
                f.write(name + "\n")
        print(f"✔ classes.txt 已生成：{classes_path}\n")
    else:
        print("✔ classes.txt 已存在，无需重复生成\n")


# ----------------------------------
# 自动标注函数
# split_name = train / val / test
# ----------------------------------
def auto_label(split_name):
    image_dir = os.path.join(BASE_IMAGE_DIR, split_name)
    label_dir = os.path.join(BASE_LABEL_DIR, split_name)
    vis_dir   = os.path.join(BASE_VIS_DIR, split_name)

    # 创建必要目录
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 检查图片目录
    if not os.path.exists(image_dir):
        print(f"❌ 图片目录不存在: {image_dir}")
        return

    # 清空 runs/detect
    clear_runs()

    # YOLO 检测
    print(f"🔍 正在使用 YOLOv8 对 {split_name} 自动标注...")
    model = YOLO("yolov8s.pt")

    model.predict(
        source=image_dir,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/detect",
        name="auto_label",
        exist_ok=True
    )
    print("✔ 自动标注完成\n")

    # 复制标签
    print(f"📂 正在复制标签到 {label_dir} ...")
    if not os.path.exists(YOLO_TEMP_LABEL_DIR):
        print("❌ 未找到 YOLO 输出标签文件")
        return

    count = 0
    for file in os.listdir(YOLO_TEMP_LABEL_DIR):
        if file.endswith(".txt"):
            shutil.copy(os.path.join(YOLO_TEMP_LABEL_DIR, file),
                        os.path.join(label_dir, file))
            count += 1

    print(f"✔ 已复制 {count} 个标签文件\n")

    # 复制可视化图片
    vis_src = "runs/detect/auto_label"
    print("🖼 正在保存可视化检测图像 …")
    for file in os.listdir(vis_src):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(os.path.join(vis_src, file), os.path.join(vis_dir, file))
    print(f"✔ 可视化图片已保存到 {vis_dir}\n")

    # 生成 classes.txt
    generate_classes_txt(label_dir)

    print(f"🎉 {split_name} 标注任务完成！\n")


# ----------------------------------
# 主程序：选择 train / val / test
# ----------------------------------
if __name__ == "__main__":
    print("请选择要自动标注的图片集：")
    print("1 - train")
    print("2 - val")
    print("3 - test")

    choice = input("请输入编号：").strip()

    if choice == "1":
        auto_label("train")
    elif choice == "2":
        auto_label("val")
    elif choice == "3":
        auto_label("test")
    else:
        print("❌ 输入错误，请输入 1 / 2 / 3")
