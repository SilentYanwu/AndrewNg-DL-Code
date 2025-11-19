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
    
    # 修正模型路径
    model_path = "runs/train/yolo_exp/weights/best.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在 - {model_path}")
        print("请检查路径是否正确，或先完成训练")
        return
    
    print(f"✅ 加载模型: {model_path}")
    model = YOLO(model_path)

    # 测试集路径
    test_source = "datasets/images/test"
    
    # 检查测试集是否存在
    if not os.path.exists(test_source):
        print(f"❌ 错误: 测试集路径不存在 - {test_source}")
        return
    
    print("🚀 开始测试集推理...")
    
    # 对测试集进行推理
    results = model.predict(
        source=test_source,
        imgsz=640,
        device=0,
        save=True,      # 保存可视化图像
        save_txt=True,  # 保存预测标签
        save_conf=True, # 保存置信度分数
        project="runs/test",
        name="exp_yolo",
        exist_ok=True
    )
    
    print(f"✅ 测试完成！结果保存在: runs/test/exp_yolo")
    
    # 可选：打印一些统计信息
    if results and len(results) > 0:
        print(f"📊 处理了 {len(results)} 张测试图片")

if __name__ == '__main__':
    main()