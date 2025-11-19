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

# 需要修复的 labels 目录
LABEL_DIRS = [
    "datasets/labels/train",
    "datasets/labels/val",
    "datasets/labels/test"
]

def fix_yolo_labels(label_dir):
    print(f"\n🔧 正在修复目录：{label_dir}")

    if not os.path.exists(label_dir):
        print(f"⚠️ 目录不存在，跳过：{label_dir}")
        return

    fixed_files = 0
    skipped_files = 0

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        if file == "classes.txt":
            continue

        path = os.path.join(label_dir, file)
        new_lines = []

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            # 仅保留前 5 列（class、x、y、w、h）
            if len(parts) >= 5:
                clean = parts[:5]
                new_lines.append(" ".join(clean))
            else:
                print(f"⚠️ 文件 {file} 中存在异常行：{line.strip()}")
                skipped_files += 1

        # 写回修复后的文件
        with open(path, "w") as f:
            for nl in new_lines:
                f.write(nl + "\n")

        fixed_files += 1
        print(f"✔ 修复成功：{file}")

    print(f"📌 修复完成：{label_dir}")
    print(f"✔ 修复文件数：{fixed_files}")
    print(f"⚠ 异常行数：{skipped_files}\n")


if __name__ == "__main__":
    print("🚀 YOLO 标签格式修复脚本开始运行...\n")
    for ld in LABEL_DIRS:
        fix_yolo_labels(ld)

