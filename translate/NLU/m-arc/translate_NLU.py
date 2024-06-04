from utils import RequestPool, check_trunk, quoter
import json
from concurrent.futures import as_completed
import yaml
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filter_file', type=str, default='./data1/language_agnostic/filter_words.yml')
parser.add_argument("--worker_num", type=int, default=30)
#parser.add_argument('--input_file', type=str, default='./UltraEval/datasets/m-mmlu/data/en.jsonl',)
parser.add_argument('--input_file', type=str, default='./UltraEval/datasets/m-arc/test.jsonl',)
parser.add_argument('--output_file', type=str, default='./UltraEval/datasets/m-arc/test_sample.jsonl',)

args = parser.parse_args()
worker_num = args.worker_num
input_path = args.input_file
output_path = args.output_file
filter_path = args.filter_file

with open(filter_path, 'r') as file:
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

def load_prompts_from_yaml(file_path, src_lang="en", target_lang="ja"):
    # 从YAML文件加载指定语言对的提示
    language_pair = f"{src_lang}-{target_lang}"
    with open(file_path, 'r', encoding='utf-8') as f:
        all_prompts = yaml.safe_load(f)
    return all_prompts.get(language_pair, {})

def translate_with_request_pool(request_pool, text, text_type, src_lang="en", target_lang="ja", additional_context=None):
    # 从YAML文件加载特定语言对的提示
    prompts = load_prompts_from_yaml('translate.yaml', src_lang, target_lang)

    # 根据text_type检索系统和用户提示
    system_prompt = prompts[text_type]["system_prompt"]
    user_prompt_preamble = prompts[text_type]["user_prompt"]
    user_prompt_translate = prompts["translate"]

    # 加载问题和答案的提示前言
    question_preamble = prompts["question_to_translate"]
    answer_preamble = prompts["answer"]

    # 构建用户提示，先发user_prompt_preamble
    combined_user_prompt = f"{user_prompt_preamble}"
    
    # 如果提供了附加上下文，则先发question_preamble，然后添加附加上下文，对附加上下文使用quoter函数
    if additional_context:
        combined_user_prompt += f"\n{question_preamble}\n{additional_context}\n\n{answer_preamble}\n"

    # 添加answer_preamble和实际的text，对text使用quoter函数
    combined_user_prompt += f"{quoter(text, 'text')}\n\n\n{user_prompt_translate}\n"
    #print(combined_user_prompt )
    # 提交翻译请求，包括系统提示和结合用户提示
    future = request_pool.commit((system_prompt, combined_user_prompt))
    return future

def write_valid_item(item, output_file_path):
    """将单个有效项写入输出文件。"""
    with open(output_file_path, 'a', encoding='utf-8') as f_out:  # 以追加模式打开
        item.pop('skip', None)
        #item['question'] += "\n答案：\n"
        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

def clean_output(out_str):
    # 清洗掉原翻译标签
    # patterns = ['<text>', '<\text>', '<\\text>']
    patterns = ['<text>','<\text>','<\\text>','</text>']


    for pattern in patterns:
        out_str = out_str.replace(pattern, '') # Replace found patterns with an empty string
    return out_str.strip()

def main(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line.strip()) for line in f]

    request_pool = RequestPool(worker_num)
    future_to_item_and_field = {}

    # 初始时确保输出文件为空
    open(output_file_path, 'w').close()

    # 第一步：首先翻译所有的'question'
    for item in items:
        item['skip'] = False  # 初始化时，没有条目被跳过

        # 检查是否需要跳过翻译
        #if not item['question'].strip() or len(item['question'].split()) > 1500:
        if not item['question'].strip() or check_trunk(item['question'], 2000) or contains_filter_word(item['question'], filter_words):
            item['skip'] = True  # 标记此条目需要跳过
            print("skip")
            continue  # 跳过当前条目的翻译过程


        # 翻译 'question' 字段
        future = translate_with_request_pool(request_pool, item['question'], 'question')
        future_to_item_and_field[future] = (item, 'question')
        
    # 等待所有'question'翻译完成
    for future in as_completed(future_to_item_and_field):
        item, field = future_to_item_and_field[future]
        try:
            translated_text = future.result()
            if translated_text == [] or len(translated_text) <= 10 or check_trunk(translated_text, 2000):
                item[field] = None  # 将项目标记为无效
            else:
                item[field] = clean_output(translated_text)  # 用翻译文本更新'question'
        except Exception as exc:
            print(f"Generated an exception: {exc}")
            item[field] = None

    future_to_item_and_field.clear() # 重置映射以进行下一批处理
    for item in items:
        # 检查跳过标志和有效性
        if item['skip'] or item['question'] is None or not item['target_scores']:
            print("skip")
            continue
        
        # 拼接选项
        options = item['target_scores']
        options_str = "\n".join([f"{chr(65+i)}.{option}" for i, option in enumerate(options)])
        # 使用翻译后的'question'作为附加上下文来翻译拼接后的选项字符串
        future = translate_with_request_pool(request_pool, options_str, 'target_scores', additional_context=item['question'])
        future_to_item_and_field[future] = (item, 'target_scores')



    # 处理'target_scores'翻译
    for future in as_completed(future_to_item_and_field):
        item, field = future_to_item_and_field[future]
        try:
            translated_text = future.result()
            if not translated_text.strip() or check_trunk(translated_text, 2000) or contains_filter_word(item['question'], filter_words):
                item[field] = None  # 将项目标记为无效
            else:
                # 清理输出文本
                translated_text = clean_output(translated_text)

                # 拆解翻译后的选项，去除额外的空格和分号，并确保正确拆分
                translated_options = [re.sub(r'^[A-Z]\.\s*', '', opt.strip()) for opt in re.split('[\n]', translated_text)]
                
                # 检查翻译后的选项是否为空，以及数量是否匹配
                if not all(translated_options) or len(translated_options) != len(item['target_scores']):
                    item['target_scores'] = None
                else:
                    # 创建新的 target_scores 字典
                    item['target_scores'] = {new_key: item['target_scores'][old_key] for new_key, old_key in zip(translated_options, item['target_scores'].keys())}



        except Exception as exc:
            print(f"Generated an exception: {exc}")
            item[field] = None  # 在异常情况下将字段标记为无效

        # 检查两次翻译是否都有效，如果有效，则写入输出文件
        if item['question'] is not None and item['target_scores'] is not None:
            write_valid_item(item, output_file_path)



# 执行主函数
if __name__ == "__main__":

    main(input_path, output_path)