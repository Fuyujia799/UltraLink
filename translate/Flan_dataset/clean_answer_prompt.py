import json

def clean_jsonl_entries(input_file_path, output_file_path):
    # 定义一个清洗字符串的函数
    def clean_input(input_str):
        # 定义需要替换为空字符串的模式列表
        patterns = ['Answer:', 'A:', 'The answer to this question is:', 'The answer is']
        for pattern in patterns:
            input_str = input_str.replace(pattern, '')  # 替换找到的模式为空字符串
        return input_str.strip()

    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            entry = json.loads(line)
            if 'inputs' in entry:
                # 使用clean_input函数清洗'inputs'字段
                entry['inputs'] = clean_input(entry['inputs'])
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')  # 为了保持jsonl格式，每个json对象后加换行符

# 使用示例
input_file_path = '/home/fuyujia/Flan_dataset/1000sample.jsonl'
output_file_path = '/home/fuyujia/Flan_dataset/1000sample_clean.jsonl'
clean_jsonl_entries(input_file_path, output_file_path)


