import pandas as pd
import matplotlib.pyplot as plt
import re

# 读取CSV文件
file_path = './attractiveness_scores.csv'
data = pd.read_csv(file_path)

# 提取文件名中的序号
data['index'] = data['filename'].apply(lambda x: int(re.search(r'\d+', x).group()))

# 按序号排序
data = data.sort_values('index')

# 绘制曲线图
plt.figure(figsize=(12, 8))
plt.plot(data['index'], data['attractiveness_score'], marker='x', linestyle='-', color='royalblue', linewidth=2, markersize=5)
# 添加标签和标题
plt.xlabel('Index in Filename', fontsize=14)
plt.ylabel('Attractiveness Score', fontsize=14)
plt.title('Attractiveness Scores vs. Index', fontsize=16)
# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)
# 设置刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 保存图片
plt.savefig("image.png")
plt.show()
