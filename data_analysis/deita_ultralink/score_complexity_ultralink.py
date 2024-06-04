import os
import json
import random
import csv
import re
from deita.selection.scorer import Llama_Scorer

model_name_or_path = "./data_analysis/deita/deita-complexity-scorer"
scorer = Llama_Scorer(model_name_or_path, is_vllm = True)

sample_size = 10000 # Set the sample size globally
csv.field_size_limit(2147483647) 

def reservoir_sampling(stream, k):
    reservoir = []
    count = 0
    for i, element in enumerate(stream):
        count += 1
        if count <= k:
            reservoir.append(element)
        else:
            probability = k / (count + 1)
            if random.random() < probability:
                reservoir[random.choice(range(k))] = element
    return reservoir

def load_json_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def load_csv_data(data_path):
    csv.field_size_limit(2147483647)  # Set field size limit to maximum
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return [''.join(row) for row in reader]

def load_text_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def process_aya_data(data):
    inputs = data.get('inputs', '')
    targets = data.get('targets', '')
    return inputs + " " + targets

def process_ultralink_data(data):
    data_content = data.get('data', '')
    if isinstance(data_content, list):
        # 拼接列表中的所有字符串
        content = " ".join(data_content)
        # 使用正则表达式查找第二个<document>之后的内容
        match = re.search(r'<document>(.*?)</document>', content, re.DOTALL)
        if match:
            # Return content after the last </document>
            last_document_index = content.rfind('</document>')  # Find the index of last </document> tag
            content = content[last_document_index + len('</document>'):]  # Extract content after the last </document>
            content = content.strip() if isinstance(content, str) else ''
            #print(content)
            return content
        else:
            # 如果没有找到第二个<document>，返回空字符串
            return content.strip() if isinstance(content, str) else ''
    else:
        # 如果data_content不是列表，则直接返回空字符串
        return ''

def process_multialpaca_data(data):
    return ''.join(data)

def process_okapi_data(data):
    combined_text = data.get("instruction", "") + " " + data.get("input", "") + " " + data.get("output", "")
    return combined_text.strip()



def process_data(data, dataset_type):
    if dataset_type == "aya":
        return process_aya_data(data)
    elif dataset_type == "ultralink":
        return process_ultralink_data(data)
    elif dataset_type == "multialpaca":
        return process_multialpaca_data(data)
    elif dataset_type == "okapi":
        return process_okapi_data(data)
    else:
        raise ValueError("Unsupported dataset type")

def run_analysis(dataset_type):
    if dataset_type == "aya":
        data_path = '/data/public/wangshuo/UltraLink/other_datas/aya_collection'
    elif dataset_type == "ultralink":
        data_folder = '/data/public/wangshuo/UltraLink/generated_datas/ultralink'
        data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.jsonl')]
    elif dataset_type == "multialpaca":
        #data_path = '/data/public/wangshuo/UltraLink/other_datas/guanaco_data/guanaco_data.csv'
        #data_path = '/data/public/wangshuo/UltraLink/other_datas/phoenix_data/phoenix_data.csv'
        data_path = '/data/public/wangshuo/UltraLink/other_datas/multialpaca/multialpaca.csv'


        data_files = [data_path]
        dataset_type = "multialpaca"
    elif dataset_type == "okapi":
        data_folder = '/data/public/wangshuo/okapi/datasets/multilingual-alpaca-52k'
        data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.json')]
    else:
        print("Unsupported dataset type")
        return

    complexity_scores = []

    for data_path in data_files:
        if dataset_type == "multialpaca":
            data = load_csv_data(data_path)
            data = reservoir_sampling(data, sample_size)  # Randomly select sample_size elements
        elif dataset_type == "okapi":
            with open(data_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                if isinstance(json_data, list):
                    data = reservoir_sampling(json_data, sample_size)
                else:
                    print("Error: Data is not in expected format.")
                    continue
        else:
            data = load_json_data(data_path)
            data = reservoir_sampling(data, sample_size)

            
        for item in data:

            processed_text = process_data(item, dataset_type)
            #print(processed_text)
            if not processed_text or len(processed_text.split()) == 0:
                continue
            complexity_score = scorer.infer_complexity(processed_text.split())

            #complexity_score = mtld(processed_text.split())
            complexity_scores.append(complexity_score)

    if complexity_scores:
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        print('Average complexity score:', avg_complexity)
    else:
        print('No valid data found.')

# Example usage
dataset_type = "multialpaca"  # Change to "aya", "ultralink", "multialpaca", or "okapi" for different datasets
run_analysis(dataset_type)
