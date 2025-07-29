"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# === 配置路径 ===
baseline_path = "./logs/ppl_baseline.csv"  # 修改成你的文件路径
ppo_path = "./logs/ppl_ppo.csv"            # 修改成你的文件路径

# === 读取数据 ===
baseline = pd.read_csv(baseline_path)
ppo = pd.read_csv(ppo_path)

# === 绘图（直接使用原始数据，不做平滑） ===
plt.figure(figsize=(8, 5))
plt.plot(baseline['step'], baseline['ppl'], label="Baseline", linewidth=2)
plt.plot(ppo['step'], ppo['ppl'], label="PPO", linewidth=2)

plt.xlabel("Training Step")
plt.ylabel("PPL (Perplexity)")
plt.title("Baseline vs PPO PPL Comparison (Raw Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === 保存图片到桌面 ===
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "ppl_comparison_raw.png")
plt.savefig(desktop_path, dpi=300)
plt.show()

print(f"图像已保存到: {desktop_path}")
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# === 配置路径 ===
baseline_path = "./logs/ppl_baseline.csv"
ppo_path = "./logs/ppl_ppo.csv"
a2c_path = "./logs/ppl_a2c.csv"

def load_clean(path):
    df = pd.read_csv(path)
    # 确保列存在
    assert {'step','ppl'}.issubset(df.columns), f"CSV 缺少列: {path}"
    # 排序并去重（同一步只保留最后一个）
    df = df.sort_values('step').drop_duplicates(subset='step', keep='last')
    return df

# === 读取并清洗数据 ===
baseline = load_clean(baseline_path)
ppo = load_clean(ppo_path)
a2c = load_clean(a2c_path)

# === 绘图（原始数据，不平滑） ===
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

# === 保存图片到桌面 ===
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "ppl_comparison_raw.png")
plt.savefig(desktop_path, dpi=300)
plt.show()

print(f"图像已保存到: {desktop_path}")
