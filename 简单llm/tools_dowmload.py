import os
import sys
import requests
import glob


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


url = "https://hf-mirror.com/datasets/zxbsmk/webnovel_cn/resolve/main/novel_cn_token512_50k.json?download=true"
save_path = "data/scifi-finetune.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

download_file(url, save_path)

sys.exit(0)

# 将文件夹下所有txt整合到一个txt中
def find_txt_files(directory):
    return glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)


def concatenate_txt_files(files, output_file):
    with open(output_file, 'w') as outfile:
        for file in files:
            with open(file, 'r') as infile:
                outfile.write(infile.read() + '\n')  # Adds a newline between files


directory = 'data'
output_file = 'data/scifi.txt'

#Find all .txt files
txt_files = find_txt_files(directory)

# Concatenate all found .txt files into one
concatenate_txt_files(txt_files, output_file)