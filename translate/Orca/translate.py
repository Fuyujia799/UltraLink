import json
import yaml
from utils import RequestPool, quoter, check_trunk  # Assuming these utility classes and functions are already defined
from concurrent.futures import as_completed

def load_prompts_from_yaml(file_path, src_lang="en", target_lang="zh"):
    # 从YAML文件加载指定语言对的提示
    language_pair = f"{src_lang}-{target_lang}"
    with open(file_path, 'r', encoding='utf-8') as f:
        all_prompts = yaml.safe_load(f)
    return all_prompts.get(language_pair, {})

def clean_input(input_str):
    # 清理输入字符串中的特定短语
    patterns = ['Answer:', 'A:', 'The answer to this question is:', 'The answer is:']
    for pattern in patterns:
        input_str = input_str.replace(pattern, '')  # Replace the patterns with an empty string
    return input_str.strip()

def clean_final_output(input_str):
    # 清理最终输出中的特定标签，并处理换行符
    patterns = ['<text>', '<\\text>']
    for pattern in patterns:
        input_str = input_str.replace(pattern, '')  # Remove specific patterns
    input_str = input_str.replace('\n', '')  
    return input_str.strip()

def translate_with_context(request_pool, text, previous_translations, prompt_type, index):
    # 载入提示，并根据上下文构建翻译请求
    prompts = load_prompts_from_yaml('translate.yaml', "en", "zh")
    prompt_info = prompts.get(prompt_type, {})
    
        # 如果是翻译第三个值，包含前两个翻译作为上下文
    if index == 2:  
        system_prompt = prompt_info["system_prompt"]
        user_prompt_preamble = prompt_info["user_prompt"]
        user_prompt_translate = prompt_info["user_translate"]
        
        question_preamble = prompt_info["user_q"] if "user_q" in prompt_info else "question："
        answer_preamble = prompt_info["user_a"] if "user_a" in prompt_info else "answer："
        
        additional_context = "\n".join(previous_translations)

        combined_user_prompt = f"{user_prompt_preamble}"
        if additional_context:
            combined_user_prompt += f"\n{question_preamble}\n{additional_context}\n\n{answer_preamble}\n"
            
        combined_user_prompt += f"{text}\n\n{user_prompt_translate}"

    else:
        combined_user_prompt = f"{prompt_info['user_prompt']}{text}\n\n{prompt_info['user_translate']}"
        system_prompt = prompt_info["system_prompt"]
        
    future = request_pool.commit((system_prompt, combined_user_prompt))
    return future

def main(input_file_path, output_file_path):
    request_pool = RequestPool(4)  
    
    with open(input_file_path, 'r', encoding='utf-8') as f_in, \
         open(output_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            conv = json.loads(line)
            conversations = conv.get('conversations', [])
            skip_entry = False  # Flag 确定是否跳过该条的翻译
            entry_translations = []  # 保存第一个和第二个value翻译之后的值，作为翻译第三个value的上文
            future_to_conversation = {}

            # 处理前两个values的翻译任务
            for i, conversation in enumerate(conversations[:2]):  
                text_to_translate = conversation['value']
                if check_trunk(text_to_translate, 1500):
                    print("Text exceeds the maximum allowed length (1500 characters), skipping translation.")
                    skip_entry = True
                    break
                if i == 1:
                    text_to_translate = clean_input(text_to_translate)
                prompt_type = "inputs"
                tagged_text = quoter(text_to_translate, 'text')
                if not skip_entry:
                    future = translate_with_context(request_pool, tagged_text, [], prompt_type, i)  # No context needed for first two
                    future_to_conversation[future] = (conversation, i)
            
            if skip_entry:
                continue

            # 等待前两个values的翻译完成
            for future in as_completed(future_to_conversation):
                conversation, index = future_to_conversation[future]
                try:
                    translated_text = future.result()
                    if not check_trunk(translated_text, 10):
                        print("Translation result has fewer than 15 tokens, skipping this entry.")
                        skip_entry = True
                        break
                    translated_text = clean_final_output(translated_text)
                    entry_translations.append(translated_text)  
                    conversation['value'] = translated_text
                except Exception as exc:
                    print(f"Translation error: {exc}")
                    skip_entry = True
                    break
            
            if skip_entry or len(conversations) < 3:
                continue
         
            # 现在处理第三个value的翻译
            conversation = conversations[2]
            text_to_translate = conversation['value']
            
            if check_trunk(text_to_translate, 1500):
                print("The third value exceeds the maximum allowed length (1500 characters), skipping translation.")
                skip_entry = True
                continue
            
            prompt_type = "targets"
            tagged_text = quoter(text_to_translate, 'text')
            future = translate_with_context(request_pool, tagged_text, entry_translations, prompt_type, 2)
            try:
                translated_text = future.result()
                if not check_trunk(translated_text, 15):
                    print("Translation result has fewer than 15 tokens, skipping this entry.")
                    continue
                translated_text = clean_final_output(translated_text)
                conversation['value'] = translated_text
            except Exception as exc:
                print(f"Translation error: {exc}")
                continue
           
            # 确认不需要跳过条目且至少有两个values时，对第二个value进行特殊处理
            if not skip_entry and len(conversations) >= 2:
                # 对第二个value的翻译结果尾部添加 "\n答案：\n"
                conversations[1]['value'] += "\n答案：\n"
            
            # 将完整的条目写入输出文件
            if not skip_entry:
                f_out.write(json.dumps(conv, ensure_ascii=False) + '\n')

                
input_file_path = './Orca/sample.jsonl'
output_file_path = './Orca/sample_translate.jsonl'

main(input_file_path, output_file_path)
