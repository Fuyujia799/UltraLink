def filter_jsonl(input_file, output_file):
    """
    筛选出不包含"English"的JSONL文件条目。

    参数:
    - input_file: 原始JSONL文件的路径。
    - output_file: 筛选后的JSONL文件的写入路径。
    """
    # 尝试打开输入文件和输出文件
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            # 逐行读取
            for line in infile:
                # 如果当前行不包含"English"，则将其写入到输出文件中
                if "english" not in line.lower():
                    outfile.write(line)
            print(f"文件已成功处理并保存到：{output_file}")
    except FileNotFoundError as e:
        print(f"文件未找到错误：{e}")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")

# 设置原始文件和目标文件的路径
input_path = '/home/fuyujia/Orca/oo-labeled_correct.gpt4.sharegpt.jsonl'
output_path = '/home/fuyujia/Orca/oo-labeled_correct.gpt4.sharegpt_no_en.jsonl'

# 调用函数进行处理
filter_jsonl(input_path, output_path)