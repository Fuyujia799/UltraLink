import ijson
import yaml
import json
import random
import os
import re
import time
import argparse
import copy
from utils import RequestPool, quoter
from concurrent.futures import as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--volume", type=int, default=10)
parser.add_argument("--worker_num", type=int, default=1000)
parser.add_argument("--en_file", type=str, default="")
parser.add_argument("--filter_file", type=str, default="")
parser.add_argument("--prompt_path" , type=str, default="./sharegpt/sharegpt_prompt.yaml")
parser.add_argument("--languages", type=str, default="zh")
parser = parser.parse_args()
# languages = ["ru", "es", "fr"]
languages = parser.languages.split(",")

languages = iter(languages)
volume = parser.volume
worker_num = parser.worker_num
en_file = parser.en_file
prompt_path = parser.prompt_path
save_path = "./sample_data/sharegpt"
os.makedirs(save_path, exist_ok=True)

with open(parser.filter_file, 'r') as file:
    filter_words_dict = yaml.safe_load(file)
    filter_words = filter_words_dict['en']
    
def contains_filter_word(data, filter_words):
    # 如果数据本身是字符串，则直接检查 
    if isinstance(data, str):
        for word in filter_words:
            pattern = r'\b' + re.escape(word) + r'\b'  # 使用单词边界确保精确匹配
            if re.search(pattern, data, re.IGNORECASE):  # 忽略大小写
                return True
    # 如果数据是字典，递归检查每个值
    elif isinstance(data, dict):
        return any(contains_filter_word(value, filter_words) for value in data.values())
    # 如果数据是列表或元组，递归检查每个元素
    elif isinstance(data, list) or isinstance(data, tuple):
        return any(contains_filter_word(item, filter_words) for item in data)
    # 其他类型的数据不包含字符串，直接返回False
    return False

def reservoir_sampling(stream, k, had_done):
    reservoir = []
    count = 0
    for i, element in enumerate(stream):
        if i in had_done or contains_filter_word(element, filter_words):
            continue
        count = count + 1
        if count <= k:
            reservoir.append(element)
        else:
            probability = k / (count + 1)
            if random.random() < probability:
                 reservoir[random.choice(range(k))] = element
    return reservoir

if __name__ == "__main__":
    for lan in languages: 
        fail_count = 0   
        out_file = os.path.join(save_path, f"sharegpt_{lan}.json")
        try:
            with open(out_file, "r") as f:
                had_done = [json.loads(line) for line in f.readlines()]
        except:
            had_done = []
        had_done = [i['id'] for i in had_done]
        with open(en_file, "r") as f:
            en_data = [json.loads(line) for line in f.readlines()]
            en_data = iter(en_data)
            sampled_data = reservoir_sampling(en_data, volume, had_done)
            en_data = iter(sampled_data)

        with open(prompt_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            for d in data:
                if d['language'] == lan:
                    prompt1 = d['prompt1']
                    prompt2 = d['prompt2']
                    text = d['text']
                    translation = d['translation']
                    break
        requestpool = RequestPool(worker_num)
        waiting_data = []
        finished_data = []
        index_list = {}
        while True:   
            for i in range(10):
                try:
                    j = next(en_data)
                except StopIteration:
                    fail_count = 1
                    break
                r = {}
                r['id'] = j['id']
                r["original_conversations"] = j["conversations"]
                r["conversations"] = copy.deepcopy(j["conversations"])
                r['futures'] = []
                for index, dialog in enumerate(r["conversations"]):
                    prompt = [prompt1, text + '\n' + dialog["value"] + "\n" + translation]
                    dialog["value"] = ""
                    future = requestpool.commit(prompt)
                    print(f"start {j['id']} {index}")
                    r['futures'].append(future)
                    index_list[future] = index
                waiting_data.append(r)
            
            for r in waiting_data:
                for future in as_completed(r['futures']):
                    index = index_list[future]
                    r['conversations'][index]['value'] = future.result()
                    print(f"finish {r['id']} {index}")
                    index_list.pop(future)
                if all([i['value'] != "" and i['value'] is not None for i in r['conversations']]):
                    del r['futures']
                    finished_data.append(r)
                else:
                    pass
            waiting_data = []
                
                
            if len(finished_data) >= 1:
                with open(out_file, "a+") as f:
                    for r in finished_data:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    finished_data = []
            
            if fail_count == 1:
                break