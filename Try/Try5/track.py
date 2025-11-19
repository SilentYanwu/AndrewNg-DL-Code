from ultralytics import YOLO
import argparse
import os
import sys

def fix_paths():
    """修复导入路径和文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

class YOLOPredictor:
    """
    YOLO预测器类 - 支持图片、视频、动图的推理和跟踪
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的模型路径 (e.g., "runs/train/yolo_exp/weights/best.pt")
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        fix_paths()
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
        
        print(f"✅ 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 支持的媒体格式
        self.supported_formats = {
            'images': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
            'videos': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'],
            'gifs': ['.gif']
        }
    
    def is_supported_file(self, file_path):
        """检查文件格式是否支持"""
        ext = os.path.splitext(file_path)[1].lower()
        all_formats = (self.supported_formats['images'] + 
                      self.supported_formats['videos'] + 
                      self.supported_formats['gifs'])
        return ext in all_formats
    
    def get_file_type(self, file_path):
        """获取文件类型"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.supported_formats['images']:
            return 'image'
        elif ext in self.supported_formats['videos']:
            return 'video'
        elif ext in self.supported_formats['gifs']:
            return 'gif'
        else:
            return 'unknown'
    
    def run_detection(self, source, output_dir="runs/detect", project_name="exp", save_txt=True):
        """
        运行目标检测
        
        Args:
            source: 输入源 (文件路径、文件夹路径、URL等)
            output_dir: 输出目录
            project_name: 项目名称
            save_txt: 是否保存标签文件
        """
        print(f"🔍 开始目标检测: {source}")
        
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device=0,  # 使用GPU，如需CPU改为 device='cpu'
            save=True,
            save_txt=save_txt,
            save_conf=True,
            project=output_dir,
            name=project_name,
            exist_ok=True
        )
        
        print(f"✅ 检测完成! 结果保存在: {output_dir}/{project_name}")
        return results
    
    def run_tracking(self, source, output_dir="runs/track", project_name="exp", tracker="botsort.yaml"):
        """
        运行目标跟踪
        
        Args:
            source: 视频源
            output_dir: 输出目录
            project_name: 项目名称
            tracker: 跟踪器配置
        """
        print(f"🎯 开始目标跟踪: {source}")
        
        results = self.model.track(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device=0,
            save=True,
            project=output_dir,
            name=project_name,
            exist_ok=True,
            tracker=tracker
        )
        
        print(f"✅ 跟踪完成! 结果保存在: {output_dir}/{project_name}")
        return results
    
    def process_directory(self, directory_path, mode="detect", output_dir=None):
        """
        处理整个目录下的文件
        
        Args:
            directory_path: 目录路径
            mode: 模式 ('detect' 或 'track')
            output_dir: 输出目录
        """
        if not os.path.exists(directory_path):
            print(f"❌ 目录不存在: {directory_path}")
            return
        
        if output_dir is None:
            output_dir = f"runs/{mode}/{os.path.basename(directory_path)}"
        
        supported_files = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path) and self.is_supported_file(file_path):
                supported_files.append(file_path)
        
        print(f"📁 找到 {len(supported_files)} 个支持的文件")
        
        for i, file_path in enumerate(supported_files, 1):
            print(f"\n📊 处理文件 {i}/{len(supported_files)}: {os.path.basename(file_path)}")
            
            file_type = self.get_file_type(file_path)
            if mode == "track" and file_type in ["video", "gif"]:
                self.run_tracking(file_path, output_dir, f"track_{i}")
            else:
                self.run_detection(file_path, output_dir, f"detect_{i}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO模型推理和跟踪工具")
    parser.add_argument("--model", type=str, required=True, help="模型路径 (e.g., runs/train/yolo_exp/weights/best.pt)")
    parser.add_argument("--source", type=str, required=True, help="输入源 (文件、目录、URL)")
    parser.add_argument("--mode", type=str, choices=["detect", "track"], default="detect", help="模式: detect(检测) 或 track(跟踪)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU阈值")
    parser.add_argument("--output", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    try:
        # 初始化预测器
        predictor = YOLOPredictor(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # 确定输出目录
        if args.output is None:
            args.output = f"runs/{args.mode}/exp"
        
        # 检查输入源类型
        if os.path.isfile(args.source):
            # 单个文件
            if args.mode == "track" and predictor.get_file_type(args.source) in ["video", "gif"]:
                predictor.run_tracking(args.source, args.output, "track_result")
            else:
                predictor.run_detection(args.source, args.output, "detect_result")
                
        elif os.path.isdir(args.source):
            # 目录
            predictor.process_directory(args.source, args.mode, args.output)
            
        else:
            # URL 或其他源
            if args.mode == "track":
                predictor.run_tracking(args.source, args.output, "track_result")
            else:
                predictor.run_detection(args.source, args.output, "detect_result")
                
        print(f"\n🎉 所有任务完成! 请检查输出目录: {args.output}")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()