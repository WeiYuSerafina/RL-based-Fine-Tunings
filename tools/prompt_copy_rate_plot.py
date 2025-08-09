import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_keywords(text):
    """Extract variable names or keywords (simple use of regular expressions)"""
    return set(re.findall(r'\b\w+\b', text))

def compute_prompt_copy_rate(prompt, generated_code):
    prompt_tokens = extract_keywords(prompt)
    gen_tokens = extract_keywords(generated_code)
    if not prompt_tokens:
        return 0.0
    copied = prompt_tokens & gen_tokens
    return len(copied) / len(prompt_tokens)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

model_files = {
     'Baseline': './baseline_generated_results.jsonl',
     'PPO': './ppo_best_step_160_generated_results.jsonl',
     'A2C': './a2c_best_step_1600_generated_results.jsonl'
}

model_copy_rates = {}

for model_name, file_path in model_files.items():
    if not os.path.exists(file_path):
        print(f"[WARNING] File does not exist: {file_path}")
        continue
    examples = load_jsonl(file_path)
    rates = []
    for item in examples:
        prompt = item.get("prompt", "")
        gen = item.get("generated_code", "")
        rate = compute_prompt_copy_rate(prompt, gen)
        rates.append(rate)
    avg_rate = np.mean(rates)
    model_copy_rates[model_name] = avg_rate
    print(f"{model_name} Average Copy Rate: {avg_rate:.4f}")

# Prepare bar chart
models = list(model_copy_rates.keys())
scores = [model_copy_rates[m] for m in models]

plt.figure(figsize=(6, 3), dpi=100)
bars = plt.bar(models, scores)
plt.ylim(0, 1)
plt.ylabel("Prompt Copy Rate")
plt.title("Prompt Copy Rate Comparison Across Models")

plt.tight_layout()
plt.savefig("prompt_copy_rate_comparison.png", dpi=300)
plt.show()
