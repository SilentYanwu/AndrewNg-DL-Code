import torch
import torch.nn as nn
import numpy as np
import time
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

# 导入自定义模块
from facenet import load_model_wrapper
from img_utils import img_to_encoding

# ---------------------------------------------------------
# 1. 配置与初始化
# ---------------------------------------------------------
def main():
    # 自动检测 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # 初始化模型并移至 GPU
    model = load_model_wrapper()
    model.to(device)
    model.eval() # 设置为评估模式 (关闭 Dropout/BN 更新)

    print("模型加载完成，参数数量:", sum(p.numel() for p in model.parameters()))

    # ---------------------------------------------------------
    # 2. Triplet Loss (PyTorch 实现)
    # ---------------------------------------------------------
    def triplet_loss(y_pred, alpha=0.2):
        """
        y_pred: (Batch, 3, 128) -> [Anchor, Positive, Negative]
        """
        anchor = y_pred[:, 0]
        positive = y_pred[:, 1]
        negative = y_pred[:, 2]

        # 计算欧氏距离平方
        pos_dist = torch.sum((anchor - positive).pow(2), dim=1)
        neg_dist = torch.sum((anchor - negative).pow(2), dim=1)

        # Loss
        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.mean(torch.clamp(basic_loss, min=0.0))
        return loss

    # ---------------------------------------------------------
    # 3. 建立人脸数据库
    # ---------------------------------------------------------
    database = {}
    # 图片存放在 data/images 目录下
    people = {
        "danielle": "data/images/danielle.png",
        "younes": "data/images/younes.jpg",
        "tian": "data/images/tian.jpg",
        "andrew": "data/images/andrew.jpg",
        "kian": "data/images/kian.jpg",
        "dan":"data/images/dan.jpg",
        "sebastiano":"data/images/sebastiano.jpg",
        "bertrand":"data/images/bertrand.jpg",
        "kevin":"data/images/kevin.jpg",
        "felix":"data/images/felix.jpg",
        "benoit":"data/images/benoit.jpg",
        "arnaud":"data/images/arnaud.jpg"
    }

    print("正在构建人脸特征数据库...")
    for name, path in people.items():
        try:
            # 传入 device 以支持 GPU 加速
            database[name] = img_to_encoding(path, model, device)
        except Exception as e:
            print(f"跳过 {name}: {e}")

    # ---------------------------------------------------------
    # 4. 验证功能 (Verify)
    # ---------------------------------------------------------
    def verify(image_path, identity, database, model):
        """
        验证图片是否属于指定身份
        """
        try:
            # 获取当前图片的编码
            encoding = img_to_encoding(image_path, model, device)
            
            # 计算与数据库中该身份的距离 (L2 Norm)
            # 注意: encoding 和 database[identity] 都是 (1, 128)
            dist = np.linalg.norm(encoding - database[identity])

            if dist < 0.7:
                print(f"[验证成功] 欢迎 {identity} 回家！ 距离: {dist:.4f}")
                is_open = True
            else:
                print(f"[验证失败] 图片与 {identity} 不符。 距离: {dist:.4f}")
                is_open = False
                
            return dist, is_open
            
        except KeyError:
            print(f"错误: 数据库中不存在用户 {identity}")
            return None, False
        except Exception as e:
            print(f"验证过程出错: {e}")
            return None, False

    # ---------------------------------------------------------
    # 5. 识别功能 (Who is it)
    # ---------------------------------------------------------
    def who_is_it(image_path, database, model):
        """
        识别图片是谁
        """
        encoding = img_to_encoding(image_path, model, device)
        
        min_dist = 100.0
        identity = None

        # 遍历数据库寻找最近邻
        for name, db_enc in database.items():
            dist = np.linalg.norm(encoding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.7:
            print(f"[未知身份] 未找到匹配的人员 (最小距离: {min_dist:.4f})")
        else:
            print(f"[识别成功] 姓名: {identity}, 距离: {min_dist:.4f}")

        return min_dist, identity

    # ---------------------------------------------------------
    # 6. 测试运行
    # ---------------------------------------------------------
    print("\n--- 开始测试 ---")
    # 确保你有这些测试图片，否则会报错
    verify("data/images/camera_0.jpg", "younes", database, model)
    who_is_it("data/images/camera_0.jpg", database, model)

if __name__ == "__main__":
    main()