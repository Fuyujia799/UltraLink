import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, stop_after_delay, 
    RetryError
)

class RequestPool:
    def __init__(self, num_workers=10):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.keys = [
            "",
        ]
        self.keys_iter = itertools.cycle(self.keys)
        self.model = "text-davinci-002"  # 使用官方文档推荐的模型
        self.api_base_url = ''
        self.clients = []
        for k in self.keys:
            client = openai.OpenAI(api_key=k)
            self.clients.append(client)
        self.clients_iter = itertools.cycle(self.clients)
    
    def commit(self, prompt):
        return self.executor.submit(self.completion_with_backoff, prompt[0], prompt[1])
    
    def submit(self, function, *args, **kwargs):
        return self.executor.submit(function, *args, **kwargs)
    
    @retry(wait=wait_random_exponential(min=1, max=5), stop=(stop_after_delay(100) | stop_after_attempt(2)))
    def completion_with_backoff(self, system_prompt, user_prompt):
        try:
            client = next(self.clients_iter)
            response = client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt,},
                    {"role": "user", "content": user_prompt,}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except KeyError:
            print("Error in message chat completions.")
            print(json.dumps(response))
            answer = ""
        except Exception as e:
            print(e)
            print("Error in message chat completions.")
            answer = ""
        return answer

    def calculate_diversity_score(self, data):
        # 获取数据的嵌入式表示
        embeddings = []
        for text in data:
            response = openai.Completion.create(
                engine=self.model,
                prompt=text
            )
            embedding = np.array(response['embedding'])
            embeddings.append(embedding)

        # 计算数据之间的余弦相似度
        similarities = cosine_similarity(embeddings)

        # 计算平均余弦相似度以衡量多样性
        average_similarity = np.mean(similarities)
        diversity_score = 1 - average_similarity

        return diversity_score

# 创建 RequestPool 实例
request_pool = RequestPool()

# 准备您的数据集
data = [
    "Your first piece of text data here",
    "Your second piece of text data here",
    "Your third piece of text data here",
    # Add more data points as needed
]

# 计算数据集的多样性分数
diversity_score = request_pool.calculate_diversity_score(data)
print("Diversity score:", diversity_score)