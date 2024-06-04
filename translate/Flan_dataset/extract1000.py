import json
import random

# Path to the original jsonl file
file_path = "/data/public/wangshuo/UltraLink/generated_datas/multi-code/MixtureCode_en.jsonl"
# Path for the new sampled jsonl file
#new_file_path = "./data1/language_agnostic/sample_data/math/math1000sample.jsonl"
new_file_path = "./data1/language_agnostic/sample_data/code/code1000sample.jsonl"

# Read the original file
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Randomly sample 1000 lines if there are more than 1000 lines
if len(lines) > 1050:
    sampled_lines = random.sample(lines, 1050)
else:
    sampled_lines = lines

# Write the sampled lines to a new file
with open(new_file_path, "w", encoding="utf-8") as new_file:
    for line in sampled_lines:
        new_file.write(line)

# Print the path of the new file
print(f"Sampled file created at: {new_file_path}")
