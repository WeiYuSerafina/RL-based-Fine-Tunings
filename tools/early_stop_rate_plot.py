import json
import os
import matplotlib.pyplot as plt

# Define your model file paths here
model_files = {
     'Baseline': './baseline_generated_results.jsonl',
     'PPO': './ppo_best_step_160_generated_results.jsonl',
     'A2C': './a2c_best_step_1600_generated_results.jsonl'
}

# Early Stop Penalty
def is_early_stopped(code: str) -> bool:
    code = code.strip()
    if "<|endoftext|>" in code:
        return True
    if len(code.split()) < 5:
        return True
    return False

# Analyse early stopping rate
model_early_stop_rates = {}

for model_name, path in model_files.items():
    if not os.path.exists(path):
        print(f"[警告] 找不到文件: {path}")
        continue

    total = 0
    early_stops = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                generated_code = obj.get("generated_code", "")
                total += 1
                if is_early_stopped(generated_code):
                    early_stops += 1
            except json.JSONDecodeError:
                continue

    rate = early_stops / total if total > 0 else 0.0
    model_early_stop_rates[model_name] = rate
    print(f"{model_name} Early Stop Rate: {rate:.2%}")

# Prepare bar chart

models = list(model_early_stop_rates.keys())
rates = [model_early_stop_rates[m] for m in models]

plt.figure(figsize=(6, 3), dpi=100)
bars = plt.bar(models, rates)
plt.ylim(0, 1)
plt.ylabel("Early Stop Rate")
plt.title("Early Stop Rate Comparison Across Models")

plt.tight_layout()
plt.savefig("early_stop_rate_comparison.png", dpi=300)
plt.show()
