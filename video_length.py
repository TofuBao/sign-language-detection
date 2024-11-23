import json
import re
from collections import Counter
import matplotlib.pyplot as plt

# 文件路径
file_path = 'C:/Users/Bill/Desktop/Ece496/data/jsonformatter.txt'

# 尝试打开文件
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: File not found at path: {file_path}")
    exit()
except json.JSONDecodeError:
    print("Error: Incorrect file format, unable to parse JSON.")
    exit()

# 存储视频长度的列表
video_lengths = []
zero_length_videos = []  # 存储视频时长为 0 秒的链接
words_in_range = []  # 存储3-3.5秒范围内的视频单词

# 提取视频长度和链接
for entry in data:
    start_time = entry.get('start_time', 0)
    end_time = entry.get('end_time', 0)
    video_length = end_time - start_time
    video_lengths.append(video_length)
    
    # 找到时长为 0 秒的视频并记录链接
    if video_length == 0:
        url = entry.get('url', 'No URL')
        zero_length_videos.append(url)
    
    # 找到时长在3-3.5秒范围内的视频单词
    if 3 <= video_length <= 3.5:
        word = entry.get('word', 'No word')
        words_in_range.append(word)

# 打印时长为 0 秒的视频链接
print("Videos with 0 second duration:")
for url in zero_length_videos:
    print(url)

# 打印时长在 3-3.5 秒范围内的视频单词
print("\nWords in videos with duration between 3 and 3.5 seconds:")
for word in words_in_range:
    print(word)

# 计算视频时长的统计信息
average_length = sum(video_lengths) / len(video_lengths) if video_lengths else 0
min_length = min(video_lengths) if video_lengths else 0
max_length = max(video_lengths) if video_lengths else 0

# 打印视频时长统计信息
print(f'\nAverage video duration: {average_length:.2f} seconds')
print(f'Shortest video duration: {min_length:.2f} seconds')
print(f'Longest video duration: {max_length:.2f} seconds')

# 绘制视频时长的分布图
plt.hist(video_lengths, bins=50, edgecolor='black')  # 根据数据自动生成bins
plt.title('Video Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.grid(True)
plt.show()
