from ultralytics import YOLO
import os
import sys

def fix_paths():
    """修复导入路径和文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

def get_user_choice():
    """获取用户选择"""
    print("\n🎯 YOLO 推理工具")
    print("=" * 30)
    
    # 选择模式
    print("\n请选择模式:")
    print("1. 🔍 检测模式 (predict)")
    print("2. 🎯 跟踪模式 (track)")
    
    while True:
        choice = input("\n请输入选择 (1 或 2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ 无效选择，请重新输入")
    
    mode = "detect" if choice == '1' else "track"
    
    # 选择文件路径
    print(f"\n请选择{'检测' if choice == '1' else '跟踪'}的文件路径:")
    print("提示: 支持图片(.jpg/.png等)、视频(.mp4/.avi等)、动图(.gif)")
    
    while True:
        file_path = input("请输入文件或目录路径: ").strip()
        
        # 处理路径中的引号
        file_path = file_path.strip('"\'')
        
        if os.path.exists(file_path):
            break
        print(f"❌ 路径不存在: {file_path}")
    
    return mode, file_path

def run_inference():
    """运行推理"""
    try:
        fix_paths()
        
        # 获取用户选择
        mode, source_path = get_user_choice()
        
        # 模型路径
        model_path = "runs/train/yolo_exp/weights/best.pt"
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请先完成训练或检查模型路径")
            return
        
        print(f"\n✅ 加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 设置输出目录
        output_dir = "runs/infer" if mode == "detect" else "runs/track"
        project_name = "result"
        
        print(f"\n🚀 开始{'检测' if mode == 'detect' else '跟踪'}...")
        print(f"📁 输入: {source_path}")
        print(f"📂 输出: {output_dir}/{project_name}")
        
        # 执行推理
        if mode == "detect":
            results = model.predict(
                source=source_path,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                device=0,
                save=True,
                save_txt=True,
                save_conf=True,
                project=output_dir,
                name=project_name,
                exist_ok=True
            )
        else:
            results = model.track(
                source=source_path,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                device=0,
                save=True,
                show=True,
                project=output_dir,
                name=project_name,
                exist_ok=True,
                tracker="botsort.yaml"
            )
        
        print(f"\n🎉 {'检测' if mode == 'detect' else '跟踪'}完成!")
        print(f"📁 结果保存在: {output_dir}/{project_name}")
        
        # 显示处理统计
        if hasattr(results, '__len__'):
            print(f"📊 处理了 {len(results)} 个文件")
        elif results:
            print("📊 处理完成")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        input("\n按回车键退出...")

if __name__ == "__main__":
    run_inference()
    input("\n按回车键退出...")