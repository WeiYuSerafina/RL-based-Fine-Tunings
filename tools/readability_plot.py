import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Define your model file paths here
model_files = {
     'Baseline': './baseline_generated_results.jsonl',
     'PPO': './ppo_best_step_160_generated_results.jsonl',
     'A2C': './a2c_best_step_1600_generated_results.jsonl'
}

# Compute readability reward
def compute_readability_reward(code: str) -> float:
    token_length = len(code.strip().split())
    return max(0.0, 1 - token_length / 100.0)

# Dictionary to hold average reward for each model
model_readability_scores = {}

# Loop through each model file
for model_name, path in model_files.items():
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        continue

    total = 0
    total_reward = 0.0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                code = obj.get("generated_code", "")
                reward = compute_readability_reward(code)
                total_reward += reward
                total += 1
            except json.JSONDecodeError:
                continue

    avg_reward = total_reward / total if total > 0 else 0.0
    model_readability_scores[model_name] = avg_reward
    print(f"{model_name} - Avg Readability Reward: {avg_reward:.4f}")

# Prepare bar chart
models = list(model_readability_scores.keys())
scores = [model_readability_scores[m] for m in models]

plt.figure(figsize=(6, 3), dpi=100)
bars = plt.bar(models, scores)
plt.ylim(0, 1)
plt.ylabel("Readability Reward")
plt.title("Readability Reward Comparison Across Models")

plt.tight_layout()
plt.savefig("readability_reward_comparison.png", dpi=300)
plt.show()
