import json
import os
import matplotlib.pyplot as plt

model_files = {
     'Baseline': './baseline_generated_results.jsonl',
     'PPO': './ppo_best_step_160_generated_results.jsonl',
     'A2C': './a2c_best_step_1600_generated_results.jsonl'
}

def is_valid_float(x):
    return isinstance(x, float) and x == x and x != float("inf") and x != float("-inf")

# simulate reward component names, you may adjust based on real data
reward_keys = ["correctness", "efficiency_reward", "readability_reward",
               "context_match_reward", "structure_reward",
               "early_stop_penalty", "shaping_bonus"]

invalid_ratio_per_model = {}

for model_name, path in model_files.items():
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        continue

    total = 0
    invalid = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                components = [data.get(k, 0.0) for k in reward_keys]
                if any(not is_valid_float(c) for c in components):
                    invalid += 1
                total += 1
            except:
                continue

    ratio = invalid / total if total > 0 else 0.0
    invalid_ratio_per_model[model_name] = ratio
    print(f"{model_name} - Invalid Reward Ratio: {ratio:.2%}")

# Plotting
models = list(invalid_ratio_per_model.keys())
ratios = [invalid_ratio_per_model[m] for m in models]

plt.figure(figsize=(6, 3), dpi=100)
bars = plt.bar(models, ratios, color="orange")
plt.ylim(0, 1)
plt.ylabel("Invalid Reward Ratio")
plt.title("Invalid Reward Component Ratio per Model")

plt.tight_layout()
plt.savefig("invalid_reward_ratio_comparison.png", dpi=300)
plt.show()
