import json
import re
from collections import Counter
import os

# 要排除的词
exclude_words = set(["org_text", "clean_text", "start_time", "signer_id", "signer", 
                     "start", "end", "file", "label", "height", "fps", "end_time", 
                     "url", "review", "text", "box"])

# 文件路径
file_path = 'C:/Users/Bill/Desktop/Ece496/data/jsonformatter.txt'
output_dir = 'C:/Users/Bill/Desktop/Ece496/data/output'  # 输出文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# 准备用于统计的词列表
word_list = []

# 提取'clean_text'字段中的词汇并进行统计
for entry in data:
    text = entry.get('clean_text', '')                                                                                          
    # 使用正则表达式去掉所有符号，只保留字母和空格，并确保只提取单独的单词
    words = re.findall(r'\b\w+\b', text.lower())
    
    # 确保每个词都是单独的，排除指定的词
    word_list.extend([word for word in words if word not in exclude_words])

# 统计词频
word_count = Counter(word_list)

# 按照出现次数从高到低排序
sorted_word_count = word_count.most_common()

# 为每个单词创建一个文件，并将相关条目保存到文件中
for word, _ in sorted_word_count:
    # 为每个词创建一个文件，文件名为单词名加上 '.txt'
    output_file_path = os.path.join(output_dir, f'{word}.txt')
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # 遍历原始数据，将 clean_text 中包含该词的条目写入文件
        for entry in data:
            clean_text = entry.get('clean_text', '').lower()
            if word in clean_text:  # 如果该条目包含该词
                output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"已为 '{word}' 生成文件并写入相关条目到 {output_file_path}")
