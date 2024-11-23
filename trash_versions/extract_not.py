import json
import os
from pytube import YouTube
from moviepy.editor import VideoFileClip

# 文件路径
file_path = 'C:/Users/Bill/Desktop/Ece496/data/jsonformatter.txt'
output_dir = 'C:/Users/Bill/Desktop/Ece496/data/videos/not'  # 下载 'not' 视频的文件夹

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 尝试打开 JSON 文件
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: 文件未找到，请检查文件路径是否正确: {file_path}")
    exit()
except json.JSONDecodeError:
    print("Error: 文件格式不正确，无法解析JSON内容。")
    exit()

# 遍历数据，下载 'not' 分类的视频并进行裁切
for entry in data:
    clean_text = entry.get('clean_text', '').lower()  # 将 clean_text 转换为小写
    if clean_text == 'not':  # 如果 'clean_text' 为 'not'
        url = entry.get('url', None)
        start_time = entry.get('start_time', 0)
        end_time = entry.get('end_time', 0)

        if url is None:
            print(f"未找到视频 URL，跳过此条目: {entry}")
            continue

        try:
            # 使用 pytube 下载视频
            yt = YouTube(url)
            video = yt.streams.filter(progressive=True, file_extension='mp4').first()
            print(f"正在下载视频: {url}")
            video_path = video.download(output_dir)  # 下载到 'not' 文件夹
            print(f"视频已下载到: {video_path}")

            # 裁切视频
            print(f"正在裁切视频，从 {start_time} 到 {end_time} 秒")
            video_clip = VideoFileClip(video_path).subclip(start_time, end_time)

            # 保存裁切后的视频
            output_cut_path = os.path.join(output_dir, f"cut_{os.path.basename(video_path)}")
            video_clip.write_videofile(output_cut_path, codec='libx264')
            print(f"裁切后的视频已保存到: {output_cut_path}")

        except Exception as e:
            print(f"处理视频时出错: {url}, 错误信息: {str(e)}")
