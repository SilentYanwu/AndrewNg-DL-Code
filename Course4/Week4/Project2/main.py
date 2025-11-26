'''
Course4.Week4.Project2 神经风格转换代码
代码来源：https://blog.csdn.net/u013733326/article/details/80767079 第二部分
和https://www.heywhale.com/mw/project/5de71d8bca27f8002c4ce1e2 
代码原使用的是 TensorFlow 1.x，需要修改为 TensorFlow 2.x
只是改得不好，似乎没有什么效果
目前已通过ai agent将代码转换到Try7（pytorch）中的main.py
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os, sys

# =========================================================
# 1. 路径修复
# =========================================================
def fix_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

fix_paths()

# =========================================================
# 2. 全局设定 - 修复参数
# =========================================================
MAX_DIM = 512
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_ITER = 200
LR = 0.02
STYLE_WEIGHT = 1e-2      # 风格权重
CONTENT_WEIGHT = 1e4     # 内容权重
TV_WEIGHT = 1e-8         # 修复：大幅降低TV权重

# =========================================================
# 3. 图像加载 / 预处理
# =========================================================
def load_img(path, max_dim=MAX_DIM):
    img = Image.open(path).convert('RGB')
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    return img

def imshow(image, title=None):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.ndim > 3:
        image = np.squeeze(image, 0)
    plt.imshow(image.astype("uint8"))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

def preprocess(image):
    return tf.keras.applications.vgg19.preprocess_input(image)

def deprocess(image):
    x = image.copy()
    if x.ndim == 4:
        x = np.squeeze(x, 0)

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]  # BGR -> RGB
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# =========================================================
# 4. 构建 VGG19 模型
# =========================================================
content_layers = ["block4_conv2"]
style_layers = [
    "block1_conv1", "block2_conv1", "block3_conv1",
    "block4_conv1", "block5_conv1"
]
num_content = len(content_layers)
num_style = len(style_layers)

def build_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    return tf.keras.Model(vgg.input, outputs)

def gram_matrix(inputs):
    b, h, w, c = inputs.shape
    features = tf.reshape(inputs, (b, h*w, c))
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(h*w*c, tf.float32)

# =========================================================
# 5. 损失函数 - 修复TV Loss计算
# =========================================================
def compute_loss(model, init_img, gram_style_targets, content_targets,
                 style_weight, content_weight, tv_weight):

    outputs = model(init_img)
    style_outputs = outputs[:num_style]
    content_outputs = outputs[num_style:]

    # Content Loss
    content_loss = tf.add_n([
        tf.reduce_mean(tf.square(c - t))
        for c, t in zip(content_outputs, content_targets)
    ]) / num_content

    # Style Loss
    style_loss = tf.add_n([
        tf.reduce_mean(tf.square(gram_matrix(o) - g))
        for o, g in zip(style_outputs, gram_style_targets)
    ]) / num_style

    # 修复：使用归一化的TV Loss
    # 将图像从VGG预处理范围转换回[0,1]范围计算TV Loss
    img_for_tv = (init_img + tf.constant([103.939, 116.779, 123.68])) / 255.0
    tv_loss = tf.image.total_variation(img_for_tv)
    tv_loss = tf.reduce_mean(tv_loss)

    total = (style_weight * style_loss +
             content_weight * content_loss +
             tv_weight * tv_loss)

    return total, style_loss, content_loss, tv_loss

# =========================================================
# 6. 单步训练
# =========================================================
@tf.function
def train_step(init_img, model, optimizer,
               gram_style_targets, content_targets,
               style_weight, content_weight, tv_weight):

    with tf.GradientTape() as tape:
        total_loss, s_loss, c_loss, tv_loss = compute_loss(
            model, init_img,
            gram_style_targets, content_targets,
            style_weight, content_weight, tv_weight
        )

    grad = tape.gradient(total_loss, init_img)
    optimizer.apply_gradients([(grad, init_img)])

    # 限制像素避免数值漂移
    init_img.assign(tf.clip_by_value(init_img, -150.0, 150.0))

    return total_loss, s_loss, c_loss, tv_loss

# =========================================================
# 7. 主训练流程
# =========================================================
def style_transfer(content_path, style_path,
                   iterations=DEFAULT_ITER,
                   style_weight=STYLE_WEIGHT,
                   content_weight=CONTENT_WEIGHT,
                   tv_weight=TV_WEIGHT,
                   lr=LR):

    print("加载图像中...")
    content = load_img(content_path)
    style = load_img(style_path)

    imshow(content, "Content")
    imshow(style, "Style")

    print("提取 VGG19 特征...")
    model = build_vgg_model()
    content_image = preprocess(content)
    style_image = preprocess(style)

    style_outputs = model(style_image)
    style_features = style_outputs[:num_style]
    gram_style_targets = [gram_matrix(f) for f in style_features]

    content_outputs = model(content_image)
    content_targets = content_outputs[num_style:]

    init_img = tf.Variable(content_image, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    print("开始训练...\n")

    for i in range(iterations + 1):
        total_loss, s_loss, c_loss, tv_loss = train_step(
            init_img, model, optimizer,
            gram_style_targets, content_targets,
            style_weight, content_weight, tv_weight
        )

        if i % 20 == 0:
            print(f"[{i:03d}] total={total_loss.numpy():.2f} "
                  f"style={s_loss.numpy():.2f} "
                  f"content={c_loss.numpy():.2f} "
                  f"tv={tv_loss.numpy():.2f}")

            out = deprocess(init_img.numpy())
            Image.fromarray(out).save(f"{OUTPUT_DIR}/step_{i}.jpg")

    final = deprocess(init_img.numpy())
    Image.fromarray(final).save(f"{OUTPUT_DIR}/final.jpg")
    print("\n最终图片已保存到 output/final.jpg\n")
    return final

# =========================================================
# 8. 运行
# =========================================================
if __name__ == "__main__":
    final = style_transfer(
        content_path="images/louvre.jpg",
        style_path="images/sandstone.jpg",
        iterations=200
    )
    plt.imshow(final)
    plt.axis("off")
    plt.show()