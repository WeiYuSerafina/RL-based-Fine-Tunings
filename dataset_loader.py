import json
import random

class MBPPDataset:
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
        sample = random.choice(self.data)
        return sample['prompt'], sample['completion']

    def lookup_ground_truth(self, prompt):
        return self.prompt2completion.get(prompt, '')


