import json
import random

class ArcadeDataset:
    def __init__(self, path):
        self.data = []
        self.prompt2completion = {}

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get('prompt', '').strip()
                completion = item.get('completion', '').strip()

                if len(prompt) > 20 and len(completion) > 10:
                    self.data.append({'prompt': prompt, 'completion': completion})
                    self.prompt2completion[prompt] = completion

    def sample(self):
        # 随机返回一个 (prompt, completion) 对
        sample = random.choice(self.data)
        return sample['prompt'], sample['completion']

    def lookup_ground_truth(self, prompt):
        # 给定 prompt，返回对应的 completion，如果找不到就返回空字符串
        return self.prompt2completion.get(prompt, '')
