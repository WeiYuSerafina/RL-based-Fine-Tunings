import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 模型对应的 JSONL 文件路径（修改为你的路径）
model_files = {
     'Baseline': './baseline_generated_results.jsonl',
     'PPO': './ppo_best_step_160_generated_results.jsonl',
     'A2C': './a2c_best_step_1600_generated_results.jsonl'
}

# 结构关键 tokens 列表
structure_tokens = ['groupby', 'mean', 'count', 'apply', 'reset_index', 'drop']

def compute_structure_reward(code: str) -> float:
    count = sum(1 for tok in structure_tokens if tok in code)
    return count / len(structure_tokens) if structure_tokens else 0.0

model_structure_scores = {}

for model_name, path in model_files.items():
    if not os.path.exists(path):
        print(f"[警告] 文件不存在: {path}")
        continue

    total = 0
    total_reward = 0.0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                code = obj.get("generated_code", "")
                reward = compute_structure_reward(code)
                total_reward += reward
                total += 1
            except json.JSONDecodeError:
                continue

    avg_reward = total_reward / total if total > 0 else 0.0
    model_structure_scores[model_name] = avg_reward
    print(f"{model_name} 平均结构性得分: {avg_reward:.4f}")

# Prepare bar chart
models = list(model_structure_scores.keys())
scores = [model_structure_scores[m] for m in models]

plt.figure(figsize=(6, 3), dpi=100)
bars = plt.bar(models, scores)
plt.ylim(0, 1)
plt.ylabel("Structure Token Usage Rate")
plt.title("Structure Token Usage Rate Comparison Across Models")

plt.tight_layout()
plt.savefig("structure_token_rate_comparison.png", dpi=300)
plt.show()
