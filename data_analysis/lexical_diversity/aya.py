import json
import random
from lexical_diversity import mtld, hdd

def reservoir_sampling(stream, k):
    reservoir = []
    count = 0
    for i, element in enumerate(stream):
        count += 1
        if count <= k:
            reservoir.append(element)
        else:
            probability = k / (count + 1)
            if random.random() < probability:
                reservoir[random.choice(range(k))] = element
    return reservoir

def calculate_scores(data_path, sample_size):
    mtld_scores = []
    hdd_scores = []

    with open(data_path, 'r', encoding='utf-8') as file:
        stream = file.readlines()

    sampled_data = reservoir_sampling(stream, sample_size)

    for line in sampled_data:
        data = json.loads(line)
        inputs = data.get('inputs', '')
        targets = data.get('targets', '')


        text = inputs + " " + targets
        # data_content = data.get('data', '')
        # #print(data_content)
        # if isinstance(data_content, list):
        #     text = " ".join(data_content)
        word_list = text.split()

        # Skip if the word list has less than 50 words
        if len(word_list) == 0:
            continue

        # Calculate MTLD score
        mtld_score = mtld(word_list)
        mtld_scores.append(mtld_score)

        # Calculate HD-D score
        # hdd_score = hdd(word_list)
        # hdd_scores.append(hdd_score)

    # Calculate average scores
    avg_mtld = sum(mtld_scores) / len(mtld_scores)
    #avg_hdd = sum(hdd_scores) / len(hdd_scores)

    return avg_mtld

data_path = './data_analysis/lexical_diversity/data/extracted_data.jsonl'
sample_size = 10000
avg_mtld = calculate_scores(data_path, sample_size)

print('Average MTLD score:', avg_mtld)
#print('Average HD-D score:', avg_hdd)
