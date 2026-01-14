import torch
from PIL import Image
from datetime import datetime
from transformers import (
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer,
    pipeline
)
import gradio as gr

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

class EnhancedImageCaptioningApp:
    def __init__(self):
        # 配置本地路径
        self.base_path = "./models"
        self.results_path = "./results"
        self.caption_model_path = os.path.join(self.base_path, "vit-gpt2")
        self.trans_model_path = os.path.join(self.base_path, "opus-mt-en-zh")
        
        # 创建必要的目录
        for path in [self.base_path, self.results_path, self.caption_model_path, self.trans_model_path]:
            os.makedirs(path, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  运行设备: {self.device.upper()}")
        
        # 初始化模型
        self.init_caption_model()
        self.init_translation_model()
        
        # 记录处理统计
        self.process_count = 0
        
    def init_caption_model(self):
        """初始化图像描述模型"""
        remote_model = "nlpconnect/vit-gpt2-image-captioning"
        
        # 检查是否需要下载模型
        model_files = os.listdir(self.caption_model_path)
        required_files = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json']
        
        if not all(f in model_files for f in required_files):
            print(f"📥 首次运行：正在下载图像描述模型到 {self.caption_model_path}...")
            try:
                model = VisionEncoderDecoderModel.from_pretrained(remote_model)
                processor = ViTImageProcessor.from_pretrained(remote_model)
                tokenizer = AutoTokenizer.from_pretrained(remote_model)
                
                model.save_pretrained(self.caption_model_path)
                processor.save_pretrained(self.caption_model_path)
                tokenizer.save_pretrained(self.caption_model_path)
                print("✅ 图像描述模型下载完成！")
            except Exception as e:
                print(f"❌ 模型下载失败: {e}")
                return
        
        print("🔧 正在加载本地图像描述模型...")
        try:
            self.model = VisionEncoderDecoderModel.from_pretrained(self.caption_model_path).to(self.device)
            self.feature_extractor = ViTImageProcessor.from_pretrained(self.caption_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.caption_model_path)
            print("✅ 图像描述模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            
    def init_translation_model(self):
        """初始化翻译模型"""
        remote_model = "Helsinki-NLP/opus-mt-en-zh"
        
        # 检查是否需要下载模型
        model_files = os.listdir(self.trans_model_path)
        
        if len(model_files) < 3:  # 简单检查
            print(f"📥 首次运行：正在下载翻译模型到 {self.trans_model_path}...")
            try:
                translator = pipeline("translation_en_to_zh", model=remote_model)
                translator.save_pretrained(self.trans_model_path)
                print("✅ 翻译模型下载完成！")
            except Exception as e:
                print(f"❌ 翻译模型下载失败: {e}")
                return
        
        print("🔧 正在加载本地翻译模型...")
        try:
            self.translator = pipeline(
                "translation_en_to_zh", 
                model=self.trans_model_path, 
                device=self.device
            )
            print("✅ 翻译模型加载成功！")
        except Exception as e:
            print(f"❌ 翻译模型加载失败: {e}")
            
    def generate_caption(self, image):
        """生成图像描述"""
        if image is None:
            return None, None, "❌ 请上传图片"
        
        try:
            # 准备图像数据
            pixel_values = self.feature_extractor(
                images=[image], 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # 生成英文描述
            output_ids = self.model.generate(
                pixel_values,
                max_length=32,
                num_beams=5,
                temperature=0.9,
                do_sample=True,
                early_stopping=True
            )
            
            caption_en = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            caption_en = caption_en.capitalize()
            
            # 翻译为中文
            if hasattr(self, 'translator'):
                translation_result = self.translator(caption_en)
                caption_zh = translation_result[0]['translation_text']
            else:
                caption_zh = "⚠️ 翻译模型未加载"
                
            # 保存结果
            save_status = self.save_results(image, caption_en, caption_zh)
            self.process_count += 1
            
            return caption_en, caption_zh, save_status
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            return None, None, error_msg
    
    def save_results(self, image, caption_en, caption_zh):
        """保存图像和描述到results目录"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"result_{timestamp}"
            
            # 保存图像
            img_path = os.path.join(self.results_path, f"{prefix}.jpg")
            image.save(img_path, "JPEG", quality=95)
            
            # 保存文本描述
            txt_path = os.path.join(self.results_path, f"{prefix}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("="*50 + "\n")
                f.write("🎨 AI 图像描述结果\n")
                f.write("="*50 + "\n\n")
                f.write(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"🖥️  处理设备: {self.device.upper()}\n")
                f.write(f"📊 处理次数: {self.process_count}\n")
                f.write("-"*50 + "\n")
                f.write("🇬🇧 英文描述 (English):\n")
                f.write(f"{caption_en}\n\n")
                f.write("🇨🇳 中文描述 (Chinese):\n")
                f.write(f"{caption_zh}\n")
                f.write("="*50 + "\n")
            
            # 返回保存状态
            status = f"✅ 结果已保存至:\n📁 {txt_path}\n🖼️  {img_path}"
            return status
            
        except Exception as e:
            return f"❌ 保存失败: {str(e)}"
    
    def get_stats(self):
        """获取处理统计信息"""
        total_results = len([f for f in os.listdir(self.results_path) 
                           if f.endswith(('.jpg', '.png', '.txt'))]) // 2
        model_size = self.get_folder_size(self.base_path)
        results_size = self.get_folder_size(self.results_path)
        
        return {
            "total_processed": self.process_count,
            "saved_results": total_results,
            "model_size": f"{model_size:.2f} MB",
            "results_size": f"{results_size:.2f} MB"
        }
    
    def get_folder_size(self, folder_path):
        """计算文件夹大小（MB）"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)

# 创建应用实例
app = EnhancedImageCaptioningApp()

# ================ Gradio 界面设计 ================
custom_css = """
:root {
    --primary-color: #4f46e5;
    --secondary-color: #7c3aed;
    --accent-color: #10b981;
    --bg-color: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
}

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.gradio-container {
    max-width: 1200px !important;
    margin: 2rem auto !important;
    border-radius: 24px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    overflow: hidden !important;
}

#main-container {
    background: var(--bg-color) !important;
    min-height: 100vh !important;
}

.header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 3rem 2rem;
    text-align: center;
    border-radius: 0 0 40px 40px;
    margin-bottom: 2rem;
}

.header h1 {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 1rem;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.header p {
    font-size: 1.2rem;
    opacity: 0.9;
    max-width: 600px;
    margin: 0 auto;
}

.card {
    background: var(--card-bg);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    margin-bottom: 2rem;
    border: 1px solid #e2e8f0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
}

.tab-nav {
    background: white;
    border-radius: 15px;
    padding: 0.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

.btn-primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4) !important;
}

.upload-area {
    border: 3px dashed #cbd5e1 !important;
    border-radius: 20px !important;
    background: #f1f5f9 !important;
    min-height: 400px !important;
}

.output-box {
    background: linear-gradient(135deg, #f8fafc, #ffffff);
    border: 2px solid #e2e8f0;
    border-radius: 15px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.stat-item {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
    border-top: 1px solid #e2e8f0;
    margin-top: 3rem;
}

/* 响应式调整 */
@media (max-width: 768px) {
    .gradio-container {
        margin: 1rem !important;
        border-radius: 16px !important;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .card {
        padding: 1.5rem;
    }
}
"""

# 构建界面
with gr.Blocks(title="🌌 离线智能图像理解系统") as demo: 
    
    with gr.Column(elem_id="main-container"):
        # 头部区域
        with gr.Column(elem_classes="header"):
            gr.Markdown("""
            # 🌌 离线智能图像理解系统
            ### Vision + Language · 全本地运行 · 隐私安全保护
            """)
        
        # 主体内容
        with gr.Tabs(elem_classes="tab-nav"):
            
            # 选项卡 1: 图像分析
            with gr.TabItem("🔮 图像理解", id=0):
                with gr.Row():
                    # 左侧上传区
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes="card"):
                            gr.Markdown("### 🖼️ 图像上传")
                            input_image = gr.Image(
                                type="pil",
                                label="拖拽图片或点击上传",
                                height=400,
                                elem_classes="upload-area"
                            )
                            
                            process_btn = gr.Button(
                                "✨ 开始智能分析",
                                variant="primary",
                                size="lg",
                                elem_classes="btn-primary"
                            )
                    
                    # 右侧结果区
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes="card"):
                            gr.Markdown("### 📝 分析结果")
                            
                            with gr.Row():
                                with gr.Column():
                                    english_output = gr.Textbox(
                                        label="🇬🇧 英文描述",
                                        placeholder="英文描述将显示在这里...",
                                        lines=4
                                    )
                            
                            with gr.Row():
                                with gr.Column():
                                    chinese_output = gr.Textbox(
                                        label="🇨🇳 中文描述",
                                        placeholder="中文翻译将显示在这里...",
                                        lines=4
                                    )
                            
                            with gr.Row():
                                save_status = gr.Textbox(
                                    label="💾 保存状态",
                                    placeholder="处理结果保存信息...",
                                    lines=3,
                                    interactive=False
                                )
            
            # 选项卡 2: 系统信息
            with gr.TabItem("📊 系统状态", id=1):
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 📈 系统统计")
                    
                    with gr.Row():
                        with gr.Column():
                            stats_output = gr.JSON(
                                label="系统统计信息",
                                value=app.get_stats()
                            )
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新统计", variant="secondary")
                    
                    gr.Markdown("### 📁 文件位置")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(f"""
                            - **🎨 模型目录**: `{app.base_path}`
                            - **💾 结果目录**: `{app.results_path}`
                            - **🔧 描述模型**: `{app.caption_model_path}`
                            - **🌐 翻译模型**: `{app.trans_model_path}`
                            """)
            
            # 选项卡 3: 使用说明
            with gr.TabItem("📚 使用指南", id=2):
                with gr.Column(elem_classes="card"):
                    gr.Markdown("""
                    ### 🚀 快速开始
                    
                    1. **上传图片**: 在"图像理解"标签页上传或拖放图片
                    2. **智能分析**: 点击"开始智能分析"按钮
                    3. **查看结果**: 系统会自动生成英文和中文描述
                    4. **结果保存**: 处理结果会自动保存到 `results/` 目录
                    
                    ### 🔧 技术特性
                    
                    - **📦 全本地运行**: 所有模型均保存在 `models/` 目录，无需网络
                    - **🔒 隐私保护**: 您的图片和描述永远不会离开您的设备
                    - **⚡ 高效推理**: 支持GPU加速，CPU模式也可流畅运行
                    - **💾 自动存档**: 所有处理结果自动保存，方便查阅
                    
                    ### 🎯 模型信息
                    
                    - **视觉模型**: ViT-GPT2 (图像描述生成)
                    - **翻译模型**: Helsinki-NLP Opus-MT (英译中)
                    - **总大小**: 约1.2GB (首次运行自动下载)
                    
                    ### ❓ 常见问题
                    
                    **Q: 首次运行需要多久？**  
                    A: 首次运行会自动下载约1.2GB模型文件，请确保网络畅通
                    
                    **Q: 支持什么格式的图片？**  
                    A: 支持JPG、PNG、BMP等常见格式，最大分辨率4096x4096
                    
                    **Q: 如何清理缓存？**  
                    A: 删除 `models/` 目录可重新下载，删除 `results/` 目录可清空历史记录
                    """)
        
        # 底部区域
        with gr.Column(elem_classes="footer"):
            gr.Markdown("""
            ---
            **🌐 离线智能图像理解系统** · 基于 ViT-GPT2 和 Opus-MT 构建  
            📅 版本 2.0 · 🔒 全本地运行 · 🚀 基于 Gradio 6.0  
            处理设备: **{}**
            """.format(app.device.upper()))
    
    # ================ 交互逻辑 ================
    
    # 图像分析功能
    process_btn.click(
        fn=app.generate_caption,
        inputs=input_image,
        outputs=[english_output, chinese_output, save_status]
    ).then(
        fn=lambda: app.get_stats(),
        outputs=stats_output
    )
    
    # 刷新统计信息
    refresh_btn.click(
        fn=lambda: app.get_stats(),
        outputs=stats_output
    )
    
    # 清除输入时重置输出
    input_image.clear(
        fn=lambda: [None, None, "📭 已清空，请上传新图片"],
        outputs=[english_output, chinese_output, save_status]
    )

# 启动应用
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 正在启动离线智能图像理解系统...")
    print("="*60)
    print("\n💡 提示:")
    print("  1. 首次运行会自动下载约1.2GB模型文件")
    print("  2. 请确保网络连接稳定")
    print("  3. 模型下载后即可完全离线使用")
    print("  4. 请访问:http://127.0.0.1:7860")
    print("="*60 + "\n")
    
    my_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate"
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        favicon_path=None,
        # 移到这里来 👇
        theme=my_theme,
        css=custom_css 
    )