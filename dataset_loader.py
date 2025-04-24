import json
import random

class ArcadeDataset:
    def __init__(self, path):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def sample(self):
        while True:
            sample = random.choice(self.data)
            prompt = sample.get('prompt', '').strip()
            completion = sample.get('completion', '').strip()

            # 过滤掉空或内容太短的样本
            if len(prompt) > 20 and len(completion) > 10:
                return prompt, completion

