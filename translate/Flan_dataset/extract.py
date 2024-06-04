import json
import os
import random

def extract_records(folder_path, output_file, record_file, num_records=1024*1024):
    # 尝试加载已有的记录文件
    try:
        with open(record_file, 'r') as f:
            extracted_records = json.load(f)
    except FileNotFoundError:
        extracted_records = {}

    new_records = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as f:
                records = [line for line in f]

            # 对于每个文件，检查是否有已抽取的记录
            file_extracted_indices = set(extracted_records.get(filename, []))
            available_indices = list(set(range(len(records))) - file_extracted_indices)

            if len(available_indices) > 0 and len(new_records) < num_records:
                needed = num_records - len(new_records)
                selected_indices = random.sample(available_indices, min(needed, len(available_indices)))
                file_extracted_indices.update(selected_indices)
                extracted_records[filename] = list(file_extracted_indices)
                
                for index in selected_indices:
                    new_records.append(records[index])
            
            # 检查是否已达到需要的记录数
            if len(new_records) >= num_records:
                break

    # 如果实际抽取到的记录数小于预期，这里可以添加一些处理逻辑
    if len(new_records) < num_records:
        print(f"Note: Only {len(new_records)} records were extracted, which is less than the requested {num_records}.")

    # 写入新的记录文件
    with open(output_file, 'w') as f:
        for record in new_records:
            f.write(record)

    # 更新记录文件
    with open(record_file, 'w') as f:
        json.dump(extracted_records, f)

# 调用函数
folder_path = './Flan_dataset/flan_filter'  # 输入文件夹路径
output_file = './Flan_dataset/flan_data_sample.jsonl'  # 输出文件路径
record_file = './Flan_dataset/record.json'  # 记录文件路径
extract_records(folder_path, output_file, record_file)
