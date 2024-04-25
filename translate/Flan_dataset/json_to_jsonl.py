import json
import os

source_folder = '/home/fuyujia/Flan_dataset/Flan_dataset'
target_folder = '/home/fuyujia/Flan_dataset/flan_jsonl'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for filename in os.listdir(source_folder):
    if filename.endswith('.json'):
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename.replace('.json', '.jsonl'))

        with open(source_file, 'r') as sf, open(target_file, 'w') as tf:
            data = json.load(sf)

            if isinstance(data, list):
                for item in data:
                    tf.write(json.dumps(item) + '\n')
            elif isinstance(data, dict):
                tf.write(json.dumps(data) + '\n')

        print(f"Completed conversion for: {filename}")

print("All files have been converted.")
