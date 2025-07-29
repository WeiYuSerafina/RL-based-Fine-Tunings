#!/usr/bin/env python
"""
plot_ppl.py – 画出 Baseline / PPO / A2C 在固定 prompt.txt 上的平滑 PPL 曲线
Author: YourName   Date: 2025-07-XX
"""
import argparse, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True,
                   help=("输入文件；可写 ① single.csv "
                         "或 ② path:MethodName 形式多个文件"))
    p.add_argument("--out", default="ppl_curve.pdf",
                   help="输出文件名（.pdf / .png）")
    p.add_argument("--title",
                   default="PPO vs A2C vs Baseline · PPL on fixed prompt.txt")
    p.add_argument("--xlabel", default="Training Step")
    p.add_argument("--ylabel", default="Perplexity ↓")
    p.add_argument("--smooth", type=int, default=5,
                   help="滚动平均窗口大小（step 数），=1 表示不平滑")
    return p.parse_args()


def load_data(csv_args):
    dfs = []
    for item in csv_args:
        if ":" in item:                  # path:Method
            path, tag = item.split(":", 1)
            df = pd.read_csv(path)
            df["method"] = tag
        else:                            # 单文件自带 method 列
            df = pd.read_csv(item)
            if "method" not in df.columns:
                raise ValueError(f"{item} 缺少 'method' 列；"
                                 "或使用 path:MethodName 语法")
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    required = {"step", "method", "ppl"}
    if not required.issubset(df_all.columns):
        raise ValueError(f"CSV 必须至少包含列 {required}")
    return df_all


def apply_smoothing(df, window):
    """对每个 method 按 step 升序做滚动平均，得到 ppl_smooth 列"""
    if window <= 1:
        df["ppl_smooth"] = df["ppl"]
        return df
    df_sorted = df.sort_values(["method", "step"])
    df_sorted["ppl_smooth"] = (
        df_sorted.groupby("method")["ppl"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    return df_sorted


def plot_ppl(df, args):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("tab10", n_colors=df["method"].nunique())

    ax = sns.lineplot(
        data=df,
        x="step",
        y="ppl_smooth",         # ← 画平滑值
        hue="method",
        estimator=None,
        ci=None,
        palette=palette,
        linewidth=2.2,
    )
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title + f"  (rolling mean = {args.smooth})")
    ax.legend(title=None, frameon=False)
    sns.despine()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"[✓] Figure saved → {args.out}")


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.csv)
    data = apply_smoothing(data, window=args.smooth)
    plot_ppl(data, args)

"""
# 如果所有方法都写在同一个 CSV（推荐）
python plot_ppl.py --csv logs/ppl_log.csv --smooth 7 \
                   --out figs/ppl_curve.pdf

# 或者分别三张 CSV，再在命令行里注明方法名
python plot_ppl.py \
       --csv baseline.csv:Baseline ppo.csv:PPO a2c.csv:A2C \
       --smooth 5 \
       --out figs/ppl_curve.pdf
"""