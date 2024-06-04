from utils import RequestPool, check_trunk, quoter
import json
from concurrent.futures import as_completed
import yaml
import re

def load_prompts_from_yaml(file_path, src_lang="en", target_lang="zh"):
    # 从YAML文件加载指定语言对的提示
    language_pair = f"{src_lang}-{target_lang}"
    with open(file_path, 'r', encoding='utf-8') as f:
        all_prompts = yaml.safe_load(f)
    return all_prompts.get(language_pair, {})

def translate_with_request_pool(request_pool, text, text_type, src_lang="en", target_lang="zh", additional_context=None):
    # 从YAML文件加载特定语言对的提示
    prompts = load_prompts_from_yaml('translate.yaml', src_lang, target_lang)

    # 根据text_type检索系统和用户提示
    system_prompt = prompts[text_type].get("system_prompt", "No system prompt available.")
    user_prompt_preamble = prompts[text_type].get("user_prompt", "No user prompt preamble available.")
    user_prompt_translate = prompts[text_type].get("user_translate", "No translate prompt available.")

    # 加载问题和答案的提示前言
    question_preamble = prompts[text_type].get("user_q", "question：")
    answer_preamble = prompts[text_type].get("user_a", "answer：")

    # 构建用户提示，先发user_prompt_preamble
    combined_user_prompt = f"{user_prompt_preamble}"
    
    # 如果提供了附加上下文，则先发question_preamble，然后添加附加上下文，对附加上下文使用quoter函数
    if additional_context:
        combined_user_prompt += f"\n{question_preamble}\n{additional_context}\n\n{answer_preamble}\n"

    # 添加answer_preamble和实际的text，对text使用quoter函数
    combined_user_prompt += f"{quoter(text, 'text')}\n\n{user_prompt_translate}"
    print(combined_user_prompt )
    # 提交翻译请求，包括系统提示和结合用户提示
    future = request_pool.commit((system_prompt, combined_user_prompt))
    return future

def write_valid_item(item, output_file_path):
    """将单个有效项写入输出文件。"""
    with open(output_file_path, 'a', encoding='utf-8') as f_out:  # 以追加模式打开
        item.pop('skip', None)
        item['inputs'] += "\n答案：\n"
        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

def clean_input(input_str):
    # 清洗掉原答案引导词
    patterns = ['Answer:', 'A:', 'The answer to this question is:', 'The answer is:']
    for pattern in patterns:
        input_str = input_str.replace(pattern, '') # Replace found patterns with an empty string
    return input_str.strip()

def clean_output(out_str):
    # 清洗掉原翻译标签
    # patterns = ['<text>', '<\text>', '<\\text>']
    patterns = ['<text>','<\\text>']

    for pattern in patterns:
        out_str = out_str.replace(pattern, '') # Replace found patterns with an empty string
    return out_str.strip()


def main(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line.strip()) for line in f]

    # Clean 'inputs' in-place before translation
    for item in items:
        if 'inputs' in item:
            item['inputs'] = clean_input(item['inputs'])

    request_pool = RequestPool(4)
    future_to_item_and_field = {}

    # 初始时确保输出文件为空
    open(output_file_path, 'w').close()

    # 第一步：首先翻译所有的'inputs'
    for item in items:
        item['skip'] = False  # 初始化时，没有条目被跳过

        # 检查是否需要跳过翻译
        #if not item['inputs'].strip() or len(item['inputs'].split()) > 1500:
        if not item['inputs'].strip() or check_trunk(item['inputs'], 1500):
            item['skip'] = True  # 标记此条目需要跳过
            print("skiplen")
            continue  # 跳过当前条目的翻译过程


        # 翻译 'inputs' 字段
        future = translate_with_request_pool(request_pool, item['inputs'], 'inputs')
        future_to_item_and_field[future] = (item, 'inputs')
        
    # 等待所有'inputs'翻译完成
    for future in as_completed(future_to_item_and_field):
        item, field = future_to_item_and_field[future]
        try:
            translated_text = future.result()
            if translated_text == [] or len(translated_text) <= 15 or check_trunk(translated_text, 2000):
                item[field] = None  # 将项目标记为无效
            else:
                item[field] = clean_output(translated_text)  # 用翻译文本更新'inputs'
        except Exception as exc:
            print(f"Generated an exception: {exc}")
            item[field] = None

    # 第二步：用翻译后的'inputs'作为上下文翻译'targets'
    future_to_item_and_field.clear()  # 重置映射以进行下一批处理
    for item in items:
        if item['skip'] or item['inputs'] is None or not item['targets'].strip():  # 检查跳过标志和有效性            
            print("skip")
            continue
                    
        # 使用翻译后的'inputs'作为附加上下文来翻译'targets'
        future = translate_with_request_pool(request_pool, item['targets'], 'targets', additional_context=item['inputs'])
        future_to_item_and_field[future] = (item, 'targets')

    # 处理'targets'翻译
    for future in as_completed(future_to_item_and_field):
        item, field = future_to_item_and_field[future]
        try:
            translated_text = future.result()
            if translated_text == [] or check_trunk(translated_text, 2000):
                item[field] = None  # 将项目标记为无效
            else:
                item[field] = clean_output(translated_text)  # 用翻译文本更新'targets'
        except Exception as exc:
            print(f"Generated an exception: {exc}")
            item[field] = None

        # 检查两次翻译是否都有效，如果有效，则写入输出文件
        if item['inputs'] is not None and item['targets'] is not None:
            write_valid_item(item, output_file_path)


# 路径
input_file_path = './Flan_dataset/en_sample.jsonl'
output_file_path = 'zh_sample.jsonl'

# 执行主函数
main(input_file_path, output_file_path)
