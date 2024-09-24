import os
import json
import string
import re
# 指定目录A的路径
directory_path = '/data/ytf/speech_llm/data/augmentation/data_aug_baseline_trans_zh'
punctuation_chars = set(string.punctuation)
chinese_punctuation_pattern = r'[{}]+'.format(''.join(['，', '。', '！', '？', '；', '：', '“', '”', '‘', '’', '&#8203;``【oaicite:0】``&#8203;', '（', '）', '《', '》']))
# 存储所有JSON数据的列表
merged_data = []

# 遍历目录A下的所有文件
for filename in os.listdir(directory_path):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每一行的JSON对象并添加到列表中
                data = json.loads(line)
                text = data.get("text", "")

                # 检查text字段的最后一个字符是否为标点符号
                if text and text[-1] in punctuation_chars:
                    merged_data.append(data)
                elif re.search(chinese_punctuation_pattern, text[-1]):
                    merged_data.append(data)

# 合并所有JSON数据
merged_json = json.dumps(merged_data, ensure_ascii=False, indent=4)

# 将合并后的JSON数据保存到文件中
output_file = '/data/ytf/speech_llm/data/augmentation/data_aug_baseline_trans_zh/combine/zh.jsonl'
with open(output_file, 'w', encoding='utf-8') as output:
    for data in merged_data:
        # 将每个JSON对象写入新文件的一行
        output.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f'合并完成，结果已保存到{output_file}')
