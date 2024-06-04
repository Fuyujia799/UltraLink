import pandas as pd
import tiktoken
from openai import OpenAI
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import os
from tqdm import tqdm
import concurrent.futures
import re
import csv
csv.field_size_limit(2147483647) 
sample_size = 10000
# Define the reservoir sampling function
def reservoir_sampling(stream, k):
    reservoir = []
    count = 0
    for i, element in enumerate(stream):
        count = count + 1
        if count <= k:
            reservoir.append(element)
        else:
            probability = k / (count + 1)
            if random.random() < probability:
                reservoir[random.choice(range(k))] = element
    return reservoir

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-",
    base_url=''
)

def load_csv_data(data_path):
    csv.field_size_limit(2147483647)  # Set field size limit to maximum
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return [''.join(row) for row in reader]
    
# Function to get text embedding
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Function to process Aya data
def process_aya_data(data):
    try:
        combined_text = data["inputs"] + " " + data["targets"]
        embedding = get_embedding(combined_text)
        return combined_text, embedding
    except Exception as e:
        #print(f"Error processing Aya data: {str(e)}")
        return None, None
    
def process_multialpaca_data(data):
    return ''.join(data)

# Function to process Ultralink data
# Function to process Ultralink data
def process_ultralink_data(data):
    try:
        if isinstance(data, str):
            data = json.loads(data)  # Convert string to dictionary
        content = data.get('data', '')
        if isinstance(content, list):
            # Concatenate all strings in the list
            content = " ".join(content)
        
        # Use regular expression to find content after the second <document>
        matches = re.findall(r'<document>(.*?)</document>', content, re.DOTALL)
        if matches:
            # Return content after the last </document>
            last_document_index = content.rfind('</document>')  # Find the index of last </document> tag
            content = content[last_document_index + len('</document>'):]  # Extract content after the last </document>
            content = content.strip() if isinstance(content, str) else ''
            #print(content)
        else:
            # If <document> tags are not found, return the entire content
            content = content.strip() if isinstance(content, str) else ''

        if content:
            embedding = get_embedding(content)
            return content, embedding
        else:
            return None, None
    except Exception as e:
        #print(f"Error processing Ultralink data: {str(e)}")
        return None, None

def process_okapi_data(data):
    try:
        content = data.get("instruction", "") + " " + data.get("input", "") + " " + data.get("output", "")
        embedding = get_embedding(content)
        return content, embedding
    except Exception as e:
        #print(f"Error processing Aya data: {str(e)}")
        return None, None

def process_multialpaca_data(data):
    try:
        content = ''.join(data)
        embedding = get_embedding(content)
        return content, embedding
    except Exception as e:
        #print(f"Error processing Aya data: {str(e)}")
        return None, None

def process_aya_collection_data(data):
    try:
        data = json.loads(data)
        inputs = data.get('inputs', '')
        targets = data.get('targets', '')
        text = inputs + " " + targets
        embedding = get_embedding(text)
        return text, embedding
    except Exception as e:
        # Handle any exceptions gracefully
        #print(f"Error processing Aya Collection data: {str(e)}")
        return None, None

# Function to calculate pairwise distances and save to CSV
def calculate_distances_and_save(data_type, data_df):
    embeddings = data_df["embedding"].tolist()
    pairwise_distances = cosine_similarity(embeddings, embeddings)
    #distance_df = pd.DataFrame(pairwise_distances, columns=range(len(data_df)), index=range(len(data_df)))
    #distance_df.to_csv(f"./data_analysis/topic_diversity/data/{data_type}_sample.csv")
    n = len(pairwise_distances)
    total_distance = sum(pairwise_distances[i][j] for i, j in combinations(range(n), 2))
    average_distance = total_distance / (n * (n - 1) / 2)
    print(f"Average cosine distance for {data_type} embeddings:", average_distance)

# Function to process data based on dataset type
# Function to process data based on dataset type
def process_data_by_type(dataset_type, data_path, executor):
    if dataset_type == "aya":
        input_datapath = data_path
        with open(input_datapath, 'r', encoding='utf-8') as file:
            stream = [json.loads(line.strip()) for line in file]  # Load JSONL lines as dictionaries
        sampled_data = reservoir_sampling(stream, 10000)
        results = list(tqdm(executor.map(process_aya_data, sampled_data), desc="Processing Aya", total=len(sampled_data)))
    
    elif dataset_type == "aya_collection":
            input_datapath = data_path
            with open(data_path, 'r', encoding='utf-8',errors = 'ignore') as file:
                stream = file.readlines()
            sampled_data = reservoir_sampling(stream, sample_size)
            results = list(tqdm(executor.map(process_aya_collection_data, sampled_data), desc="Processing Aya Collection", total=len(sampled_data)))
            
    elif dataset_type == "ultralink":
        data_files = [
            os.path.join(data_path, file)
            for file in os.listdir(data_path)
            #if file.endswith('.jsonl') and 'specific' not in file
            if file.endswith('.jsonl')

        ]
        sampled_data = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                stream = [json.loads(line.strip()) for line in file]  # Load JSONL lines as dictionaries
            sampled_data.extend(reservoir_sampling(stream, sample_size//28))
        results = list(tqdm(executor.map(process_ultralink_data, sampled_data), desc="Processing Ultralink", total=len(sampled_data)))

    elif dataset_type == "okapi":
        data_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.json')]
        sampled_data = []
                    
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                stream = json.load(file)
                #stream = [json.loads(line.strip()) for line in file]  # Load JSONL lines as dictionaries
            sampled_data.extend(reservoir_sampling(stream, sample_size//7))
        results = list(tqdm(executor.map(process_okapi_data, sampled_data), desc="Processing okapi", total=len(sampled_data)))
    
    elif dataset_type == "multialpaca":
        data = load_csv_data(data_path)
        sampled_data = reservoir_sampling(data, sample_size)  # Randomly select sample_size elements
        results = list(tqdm(executor.map(process_multialpaca_data, sampled_data), desc="Processing multialpaca", total=len(sampled_data)))

        
    else:
        raise ValueError("Invalid dataset type specified.")

    results = [result for result in results if result[0] is not None and result[1] is not None]
    combined_texts, embeddings = zip(*results)
    df = pd.DataFrame({"combined": combined_texts})
    df["embedding"] = embeddings
    return df

# Main function
def main(dataset_type):
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        if dataset_type == "aya":
            data_path = './data_analysis/lexical_diversity/data/extracted_data.jsonl'
        
        if dataset_type == "aya_collection":
            data_path = './data_analysis/lexical_diversity/data/extracted_data.jsonl'

        elif dataset_type == "ultralink":
            data_path = '/data/public/wangshuo/UltraLink/generated_datas/ultralink'
            
        elif dataset_type == "multialpaca":
            #data_path = '/data/public/wangshuo/UltraLink/other_datas/multialpaca/multialpaca.csv'
            #data_path = '/data/public/wangshuo/UltraLink/other_datas/phoenix_data/phoenix_data.csv'
            data_path = '/data/public/wangshuo/UltraLink/other_datas/guanaco_data/guanaco_data.csv'

            data_files = [data_path]
            dataset_type = "multialpaca"
            
        elif dataset_type == "okapi":
            data_path = '/data/public/wangshuo/okapi/datasets/multilingual-alpaca-52k'
            #data_path = '/data/public/wangshuo/UltraLink/other_datas/guanaco_data/guanaco_data.csv'

            #data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.json')]
        else:
            raise ValueError("Invalid dataset type specified.")
        
        data_df = process_data_by_type(dataset_type, data_path, executor)
        calculate_distances_and_save(dataset_type, data_df)

# Run the script
if __name__ == "__main__":
    dataset_type = "multialpaca"  # Change this to "aya" or "ultralink" as needed
    main(dataset_type)
