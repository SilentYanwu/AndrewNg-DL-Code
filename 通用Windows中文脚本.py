import matplotlib.pyplot as plt

# 设置 Matplotlib 使用支持中文的字体（Windows 推荐 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题