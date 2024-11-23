import json
from collections import Counter

# 要排除的词
exclude_words = set(["org_text", "clean_text", "start_time", "signer_id", "signer", 
                     "start", "end", "file", "label", "height", "fps", "end_time", 
                     "url", "review", "text", "box"])

# 文件路径
file_path = 'C:/Users/Bill/Desktop/Ece496/data/jsonformatter.txt'

# 尝试打开文件
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: 文件未找到，请检查文件路径是否正确: {file_path}")
    exit()
except json.JSONDecodeError:
    print("Error: 文件格式不正确，无法解析JSON内容。")
    exit()

# 准备用于统计的词频列表
word_count = Counter()

# 提取 'clean_text' 字段中的词汇，并确保统计唯一的词汇
for entry in data:
    clean_text = entry.get('clean_text', '').lower()  # 确保 'clean_text' 为小写
    # 如果 clean_text 是我们想要统计的单个词，并且没有在排除词列表中
    if clean_text and clean_text not in exclude_words:
        word_count[clean_text] += 1  # 每个条目只统计一次

# 按照出现次数从高到低排序
sorted_word_count = word_count.most_common()

# 打印结果
for word, count in sorted_word_count:
    print(f'{word}: {count}')
