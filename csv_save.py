import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration path
baseline_path = "./logs/ppl_baseline.csv"
ppo_path = "./logs/ppl_ppo.csv"
a2c_path = "./logs/ppl_a2c.csv"

def load_clean(path):
    df = pd.read_csv(path)
    assert {'step','ppl'}.issubset(df.columns), f"CSV missing columns: {path}"
    df = df.sort_values('step').drop_duplicates(subset='step', keep='last')
    return df

# Read and clean data
baseline = load_clean(baseline_path)
ppo = load_clean(ppo_path)
a2c = load_clean(a2c_path)

# Plot (raw data, not smoothed)
plt.figure(figsize=(8, 5))
plt.plot(baseline['step'], baseline['ppl'], label="Baseline", linewidth=2)
plt.plot(ppo['step'], ppo['ppl'], label="PPO", linewidth=2)
plt.plot(a2c['step'], a2c['ppl'], label="A2C", linewidth=2)  # ← 修正点

plt.xlabel("Training Step")
plt.ylabel("PPL (Perplexity)")
plt.title("Baseline vs PPO vs A2C PPL Comparison (Raw Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "ppl_comparison_raw.png")
plt.savefig(desktop_path, dpi=300)
plt.show()

print(f"Figure saved to: {desktop_path}")
