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

# ================ 全新大气桌面级 UI ================
custom_css = """
/* 全局样式 - 专业深色主题 */
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0c12;
    background-image: radial-gradient(circle at 10% 20%, rgba(25, 30, 45, 1) 0%, #0a0c12 100%);
    min-height: 100vh;
}

/* 主容器 - 桌面宽屏布局 */
.gradio-container {
    max-width: 1600px !important;
    margin: 2rem auto !important;
    background: rgba(15, 20, 30, 0.65) !important;
    backdrop-filter: blur(10px);
    border-radius: 2rem !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
    overflow: hidden !important;
}

/* 主内容区域 */
#main-container {
    padding: 2rem 2rem 1.5rem 2rem !important;
}

/* 头部 Hero 区域 */
.hero-section {
    text-align: center;
    margin-bottom: 2.5rem;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border-radius: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #9ca3af;
    max-width: 550px;
    margin: 0 auto;
}

/* 左右两栏布局 */
.two-columns {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}

.left-panel {
    flex: 1.2;
    min-width: 280px;
}

.right-panel {
    flex: 2;
    min-width: 400px;
}

/* 玻璃卡片 */
.glass-card {
    background: rgba(25, 32, 45, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 1.5rem;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 1.5rem;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 35px -12px rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.2);
}

.card-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    border-left: 3px solid #8b5cf6;
    padding-left: 1rem;
}

/* 上传区域 */
.upload-area {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 2px dashed #4b5563 !important;
    border-radius: 1.2rem !important;
    transition: all 0.2s;
}

.upload-area:hover {
    border-color: #8b5cf6 !important;
    background: rgba(0, 0, 0, 0.45) !important;
}

/* 按钮样式 */
.primary-btn {
    background: linear-gradient(95deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.8rem 1.8rem !important;
    border-radius: 2rem !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s !important;
    width: 100%;
}

.primary-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(139, 92, 246, 0.5) !important;
}

.secondary-btn {
    background: rgba(45, 55, 75, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e5e7eb !important;
    font-weight: 500 !important;
    border-radius: 2rem !important;
    padding: 0.5rem 1.2rem !important;
}

.secondary-btn:hover {
    background: rgba(55, 65, 85, 0.9) !important;
    border-color: #8b5cf6 !important;
}

/* 输出文本框 */
.output-textbox textarea, .output-textbox input {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 1px solid #374151 !important;
    border-radius: 1rem !important;
    color: #f3f4f6 !important;
    font-size: 0.95rem !important;
    padding: 0.8rem !important;
    font-family: 'Inter', monospace;
}

.output-textbox label {
    color: #d1d5db !important;
    font-weight: 500 !important;
    margin-bottom: 0.4rem !important;
}

/* 统计卡片网格 */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.stat-item {
    background: rgba(0, 0, 0, 0.35);
    border-radius: 1.2rem;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.stat-label {
    color: #9ca3af;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
}

/* 文件路径区域 */
.path-box {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 1rem;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: #9ca3af;
    word-break: break-all;
}

/* 底部 */
.footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    font-size: 0.8rem;
    color: #6b7280;
    border-top: 1px solid rgba(255,255,255,0.05);
}

/* 选项卡样式 */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tabs > .tab-nav {
    background: rgba(20, 28, 40, 0.7);
    backdrop-filter: blur(8px);
    border-radius: 1.2rem;
    padding: 0.3rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.tabs button {
    border-radius: 1rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    background: transparent !important;
    color: #9ca3af !important;
}

.tabs button.selected {
    background: rgba(99, 102, 241, 0.25) !important;
    color: white !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.3) !important;
}

/* 滚动条美化 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #1f2937;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
}

/* 响应式：小屏幕时变为纵向 */
@media (max-width: 900px) {
    .two-columns {
        flex-direction: column;
    }
    .left-panel, .right-panel {
        min-width: auto;
    }
    .hero-title {
        font-size: 2rem;
    }
    .gradio-container {
        margin: 1rem !important;
        border-radius: 1.5rem !important;
    }
    #main-container {
        padding: 1rem !important;
    }
}
"""

# ================ 构建 UI（保持逻辑不变） ================
with gr.Blocks(title="🌌 智能图像理解系统", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        # 头部
        with gr.Column(elem_classes="hero-section"):
            gr.HTML("""
                <div class="hero-title">🌌 智能图像理解系统</div>
                <div class="hero-subtitle">ViT-GPT2 + Opus-MT · 全本地运行 · 企业级隐私保护</div>
            """)
        
        # 选项卡（保持原有三个标签，但放在内部）
        with gr.Tabs(elem_classes="tabs"):
            # ---------- 图像理解 ----------
            with gr.TabItem("🔮 图像理解"):
                # 左右两栏布局
                with gr.Column(elem_classes="two-columns"):
                    # 左侧面板：上传和操作
                    with gr.Column(elem_classes="left-panel"):
                        with gr.Column(elem_classes="glass-card"):
                            gr.HTML('<div class="card-header">📤 上传图像</div>')
                            input_image = gr.Image(
                                type="pil",
                                label=None,
                                height=380,
                                elem_classes="upload-area"
                            )
                            process_btn = gr.Button(
                                "✨ 开始智能分析",
                                variant="primary",
                                elem_classes="primary-btn"
                            )
                    
                    # 右侧面板：结果展示
                    with gr.Column(elem_classes="right-panel"):
                        with gr.Column(elem_classes="glass-card"):
                            gr.HTML('<div class="card-header">📝 分析结果</div>')
                            with gr.Column(elem_classes="output-textbox"):
                                english_output = gr.Textbox(
                                    label="🇬🇧 英文描述 (English)",
                                    placeholder="等待分析...",
                                    lines=3,
                                    interactive=False
                                )
                            with gr.Column(elem_classes="output-textbox"):
                                chinese_output = gr.Textbox(
                                    label="🇨🇳 中文描述 (Chinese)",
                                    placeholder="等待翻译...",
                                    lines=3,
                                    interactive=False
                                )
                            with gr.Column(elem_classes="output-textbox"):
                                save_status = gr.Textbox(
                                    label="💾 保存状态",
                                    placeholder="结果保存信息",
                                    lines=2,
                                    interactive=False
                                )
            
            # ---------- 系统状态 ----------
            with gr.TabItem("📊 系统状态"):
                with gr.Column(elem_classes="glass-card"):
                    gr.HTML('<div class="card-header">📈 运行统计</div>')
                    stats_output = gr.JSON(label="系统统计信息", value=app.get_stats())
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新统计", elem_classes="secondary-btn", size="sm")
                    gr.HTML('<div class="card-header" style="margin-top: 1.5rem;">🗂️ 存储位置</div>')
                    gr.Markdown(f"""
                    <div class="path-box">
                    🎨 模型目录：`{app.base_path}`<br>
                    💾 结果目录：`{app.results_path}`<br>
                    🔧 描述模型：`{app.caption_model_path}`<br>
                    🌐 翻译模型：`{app.trans_model_path}`
                    </div>
                    """)
            
            # ---------- 使用指南 ----------
            with gr.TabItem("📚 使用指南"):
                with gr.Column(elem_classes="glass-card"):
                    gr.HTML('<div class="card-header">🚀 快速开始</div>')
                    gr.Markdown("""
                    **1. 上传图片** → 在「图像理解」标签页上传或拖拽图像  
                    **2. 智能分析** → 点击「开始智能分析」按钮  
                    **3. 查看结果** → 系统自动生成英文描述并翻译为中文  
                    **4. 自动保存** → 结果保存至 `results/` 目录，便于回顾  
                    
                    ---
                    ### 🔧 技术特性
                    - **全本地运行** – 所有模型保存在 `models/` 目录，无需互联网  
                    - **隐私安全** – 您的图像和描述永不离开本机  
                    - **GPU/CPU 自适应** – 自动选择最佳推理设备  
                    - **自动存档** – 每次处理结果自动保存为图像 + 文本文件  
                    
                    ### 🎯 模型信息
                    - **视觉模型**：ViT-GPT2（图像描述生成）  
                    - **翻译模型**：Helsinki-NLP Opus-MT（英文 → 中文）  
                    - **总大小**：约 1.2 GB（首次运行自动下载）  
                    
                    ### ❓ 常见问题
                    - **首次运行慢？** 需要下载模型，请保持网络稳定  
                    - **支持哪些图片格式？** JPG、PNG、BMP 等常见格式  
                    - **如何清理缓存？** 删除 `models/` 目录可重新下载模型，删除 `results/` 可清空历史记录
                    """)
        
        # 底部
        with gr.Column(elem_classes="footer"):
            gr.Markdown(f"""
            **离线智能图像理解系统** · 基于 ViT-GPT2 和 Opus-MT  
            🖥️ 当前设备：**{app.device.upper()}** · 🔒 全本地运行 · 🚀 Gradio 驱动
            """)
    
    # ================ 交互逻辑（完全保留） ================
    process_btn.click(
        fn=app.generate_caption,
        inputs=input_image,
        outputs=[english_output, chinese_output, save_status]
    ).then(
        fn=lambda: app.get_stats(),
        outputs=stats_output
    )
    
    refresh_btn.click(
        fn=lambda: app.get_stats(),
        outputs=stats_output
    )
    
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
    print("  4. 请访问: http://127.0.0.1:7860")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        favicon_path=None
    )