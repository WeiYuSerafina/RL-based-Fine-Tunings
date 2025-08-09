import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set file paths for each model
model_files = {
     'Baseline': './baseline_generated_results.jsonl',
     'PPO': './ppo_best_step_160_generated_results.jsonl',
     'A2C': './a2c_best_step_1600_generated_results.jsonl'
}

# Function to compute shaping bonus
def compute_shaping_bonus(code: str) -> float:
    code = code.strip()
    if code == "" or "<|endoftext|>" in code:
        return -0.5
    else:
        return 0.2

# Dictionary to hold average bonus for each model
model_shaping_scores = {}

# Loop over each model's output file
for model_name, path in model_files.items():
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        continue

    total = 0
    total_bonus = 0.0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                code = obj.get("generated_code", "")
                bonus = compute_shaping_bonus(code)
                total_bonus += bonus
                total += 1
            except json.JSONDecodeError:
                continue

    avg_bonus = total_bonus / total if total > 0 else 0.0
    model_shaping_scores[model_name] = avg_bonus
    print(f"{model_name} - Avg Shaping Bonus: {avg_bonus:.4f}")

# Create bar chart
models = list(model_shaping_scores.keys())
scores = [model_shaping_scores[m] for m in models]

plt.figure(figsize=(6, 3), dpi=100)
#bars = plt.bar(models, scores, color='skyblue')
bars = plt.bar(models, scores, color=plt.cm.tab10(0))
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel("Shaping Bonus")
plt.title("Shaping Bonus Comparison Across Models")
plt.ylim(-0.6, 0.3)

plt.tight_layout()
plt.savefig("shaping_bonus_comparison.png", dpi=300)
plt.show()
