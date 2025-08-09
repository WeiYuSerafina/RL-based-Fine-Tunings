import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
baseline_path = "./logs/ppl_baseline.csv"
ppo_path = "./logs/ppl_ppo.csv"

# Reading Data
baseline = pd.read_csv(baseline_path)
ppo = pd.read_csv(ppo_path)


# Smoothing
def smooth(y, box_pts=5):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

baseline['ppl_smooth'] = smooth(baseline['ppl'], box_pts=5)
ppo['ppl_smooth'] = smooth(ppo['ppl'], box_pts=5)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(baseline['step'], baseline['ppl_smooth'], label="Baseline (Smoothed)", linewidth=2)
plt.plot(ppo['step'], ppo['ppl_smooth'], label="PPO (Smoothed)", linewidth=2)

plt.xlabel("Training Step")
plt.ylabel("PPL (Perplexity)")
plt.title("Baseline vs PPO PPL Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "ppl_comparison.png")
plt.savefig(desktop_path, dpi=300)
plt.show()

print(f"Figure saved to: {desktop_path}")
