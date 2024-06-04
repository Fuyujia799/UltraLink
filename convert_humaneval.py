import ijson
import yaml
import json
import random
import os
import re
import argparse
from utils import RequestPool, quoter
from concurrent.futures import as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--volume", type=int, default=30)
parser.add_argument("--worker_num", type=int, default=50)
parser.add_argument("--en_file", type=str, default="")
parser.add_argument("--filter_file", type=str, default="")
parser.add_argument("--prompt_path" , type=str, default="./humaneval/prompt.yaml")
parser.add_argument("--languages", type=str, default="ru")
parser = parser.parse_args()
# languages = ["ru", "es", "fr"]
languages = parser.languages.split(",")
matcher = re.compile(r"([\"']{3}.*?[\"']{3})", re.DOTALL)

languages = iter(languages)
volume = parser.volume
worker_num = parser.worker_num
en_file = parser.en_file
prompt_path = parser.prompt_path
save_path = "./humaneval/"
os.makedirs(save_path, exist_ok=True)

with open(parser.filter_file, 'r') as file:
    filter_words_dict = yaml.safe_load(file)
    filter_words = filter_words_dict['en']

def contains_filter_word(element, filter_words):
    for value in element.values():
        if isinstance(value, str):
            for word in filter_words:
                if word in value:
                    print(f"Filtered out: {element} because of word: {word}")
                    return True
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
        out_file = os.path.join(save_path, f"humaneval_{lan}.jsonl")
        try:
            with open(out_file, "r") as f:
                had_done = [json.loads(line) for line in f.readlines()]
        except FileNotFoundError:
            had_done = []
        had_done = [i['task_id'] for i in had_done]

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
        result = []
        futures = []
        data = {}

        while len(futures) < min(worker_num, volume):
            try:
                j = next(en_data)
            except StopIteration:
                print("no data")
                fail_count = 1
                break
            r = {}
            r['task_id'] = j['task_id']
            r['prompt'] = ''
            r['entry_point'] = j['entry_point']
            r['canonical_solution'] = j['canonical_solution']
            r['test'] = j['test']
            
            prompt_content_match = matcher.search(j['prompt'])
            #print(prompt_content_match)
            if prompt_content_match:
                prompt_content = prompt_content_match.group(0)
                #print(prompt_content)
                prompt_to_translate = prompt1 + text + prompt_content + translation
                #print(prompt_to_translate)
                p = ["", prompt_to_translate]

                print(f"start {j['task_id']}")
                future = requestpool.commit(p)
                futures.append(future)  
                data[future] = (r, j, prompt_content, j['prompt'].replace(prompt_content, "{}"))
            else:
                continue  # Skip this item if no match is found
        
        while futures:
            new_futures = []
            for future in as_completed(futures):
                r, j, original_content, prompt_template = data[future]
                translated_content = future.result()  
                #print(translated_content)
                if translated_content is not None and translated_content:
                    quote_type = original_content[0]  # 获取最初匹配的引号类型
                    translated_prompt = prompt_template.format(f"{quote_type*3}" + translated_content + f"{quote_type*3}")
                    # 替换四个及以上的引号为三个引号
                    translated_prompt = re.sub(r'\"{4,}', '\"\"\"', translated_prompt)
                    translated_prompt = re.sub(r'\'{4,}', '\'\'\'', translated_prompt)
                    # 检查三个引号后是否有换行符，如果没有则添加
                    translated_prompt = re.sub(r'\"\"\"(?!\n)', '\"\"\"\n    ', translated_prompt)
                    translated_prompt = re.sub(r'\'\'\'(?!\n)', '\'\'\'\n    ', translated_prompt)
                    translated_prompt = re.sub(r'(?m)^("""|\'\'\')\n(?![ ]{4})', r'"""\n    ', translated_prompt)
                    
                    r['prompt'] = translated_prompt
                    result.append(r)
                    print(f"done {r['task_id']}")
                del data[future]
                try:
                    j = next(en_data)
                except StopIteration:
                    fail_count += 1
                    continue
                prompt_content_match = matcher.search(j['prompt'])
                if prompt_content_match:
                    prompt_content = prompt_content_match.group(0)
                    prompt_to_translate = prompt1 + text + prompt_content + translation
                    #print(prompt_to_translate)
                    p = ["", prompt_to_translate]

                    print(f"start {j['task_id']}")
                    future = requestpool.commit(p)
                    new_futures.append(future)  
                    data[future] = (r, j, prompt_content, j['prompt'].replace(prompt_content, "{}"))
            
            futures = new_futures
                
            if result:
                with open(out_file, "a+") as f:
                    for r in result:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    result = []

            if fail_count > 0:
                break

    