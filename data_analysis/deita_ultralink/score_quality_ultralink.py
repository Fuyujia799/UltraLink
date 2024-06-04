import os
import json
import random
import csv
import re
from deita.selection.scorer import Llama_Scorer

model_name_or_path = "./data_analysis/deita/deita-quality-scorer"

scorer = Llama_Scorer(model_name_or_path, is_vllm = True)

sample_size = 10000//30  # Set the sample size globally
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
    with open(data_path, 'r', encoding='utf-8', errors = 'ignore') as file:
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
    #print(inputs)
    #print("------")
    #print(targets)
    return inputs, targets

def process_ultralink_data(data):
    data_content = data.get('data', '')
    if isinstance(data_content, list):
        # Extract content after </document> in the first item and all odd-indexed items
        input_text = ""
        output_text = ""
        for index, item in enumerate(data_content):
            if index == 0:
                # Get content after </document> in the first item
                match = re.search(r'</document>(.*?)$', item, re.DOTALL)
                if match:
                    input_text += match.group(1).strip() if match else ''
                else:
            # If <document> tags are not found, return the entire content
                    input_text += item.strip() if isinstance(item, str) else ''
            elif index == 1:
                # Odd-indexed items
                output_text += ' ' + item
            else:
                # Even-indexed items
                #output_text += ' ' + item
                input_text += ''
        #print(input_text.strip())
        #print(1)
        #print(output_text.strip())
        return input_text.strip(), output_text.strip()
    else:
        return '', ''


def process_multialpaca_data(data):
    split_data = data.split("'")
    first_content = split_data[1] if len(split_data)>1 else None
    third_content = split_data[3] if len(split_data)>3 else None
    #print(first_content)
    #print(third_content)
    return first_content, third_content



def process_okapi_data(data):
    instruction = data.get("instruction", "")
    input_text = data.get("input", "")
    output_text = data.get("output", "")
    #print(instruction + " " + input_text)
    #print(output_text)
    return instruction + " " + input_text, output_text


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
        #data_path = '/data/public/wangshuo/UltraLink/other_datas/aya_collection/aya_dataset_train.jsonl'
        data_path = './data_analysis/lexical_diversity/data/extracted_data.jsonl'
        data_files = [data_path]
    elif dataset_type == "ultralink":
        data_folder = '/data/public/wangshuo/UltraLink/generated_datas/ultralink'
        data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.jsonl')]
    elif dataset_type == "multialpaca":
        #data_path = '/data/public/wangshuo/UltraLink/other_datas/guanaco_data/guanaco_data.csv'
        data_path = '/data/public/wangshuo/UltraLink/other_datas/phoenix_data/phoenix_data.csv'
        #data_path = '/data/public/wangshuo/UltraLink/other_datas/multialpaca/multialpaca.csv'
        data_files = [data_path]
        dataset_type = "multialpaca"
    elif dataset_type == "okapi":
        data_folder = '/data/public/wangshuo/okapi/datasets/multilingual-alpaca-52k'
        data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.json')]
    else:
        print("Unsupported dataset type")
        return

    quality_scores = []

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
            input_text, output_text = process_data(item, dataset_type)
            if not input_text or not output_text:
                continue
            quality_score = scorer.infer_quality(input_text, output_text)
            quality_scores.append(quality_score)

    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        print('Average quality score:', avg_quality)
    else:
        print('No valid data found.')


# Example usage
dataset_type = "ultralink"  # Change to "aya", "ultralink", "multialpaca", or "okapi" for different datasets
run_analysis(dataset_type)
