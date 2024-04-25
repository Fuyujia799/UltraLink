import json

input_file = '/home/fuyujia/UltraEval/datasets/m-hellaswag/test_sample.jsonl'
output_file = '/home/fuyujia/UltraEval/datasets/m-hellaswag/ja.jsonl'

with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as fout:
    for line in f:
        # 解析每一行的JSON数据
        data = json.loads(line)

        # 检查并修改'question'字段
        if data['question'].endswith('...'):
            data['question'] = data['question'][:-3]  # 删除末尾的"..."

        # 将修改后的数据写回到新文件
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
