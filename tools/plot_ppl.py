"""
plot_ppl.py – Plot smoothed PPL curves of Baseline / PPO / A2C on fixed prompt.txt
Author: YourName   Date: 2025-07-XX
"""
import argparse, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True,
                   help=("Input CSV file(s). Use either: "
                         "① a single file with a 'method' column, or "
                         "② multiple files in the format path:MethodName"))
    p.add_argument("--out", default="ppl_curve.pdf",
                   help="Output filename (.pdf / .png)")
    p.add_argument("--title",
                   default="PPO vs A2C vs Baseline · PPL on fixed prompt.txt")
    p.add_argument("--xlabel", default="Training Step")
    p.add_argument("--ylabel", default="Perplexity ↓")
    p.add_argument("--smooth", type=int, default=5,
                   help="Rolling average window size (in steps), set to 1 to disable smoothing")
    return p.parse_args()


def load_data(csv_args):
    dfs = []
    for item in csv_args:
        if ":" in item:  # path:Method format
            path, tag = item.split(":", 1)
            df = pd.read_csv(path)
            df["method"] = tag
        else:  # file should already contain 'method' column
            df = pd.read_csv(item)
            if "method" not in df.columns:
                raise ValueError(f"{item} is missing a 'method' column; "
                                 "or use the path:MethodName format")
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    required = {"step", "method", "ppl"}
    if not required.issubset(df_all.columns):
        raise ValueError(f"CSV must contain the following columns: {required}")
    return df_all


def apply_smoothing(df, window):
    """Apply rolling mean smoothing per method over step"""
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
        y="ppl_smooth",         # plot smoothed value
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
# Example 1: If all methods are in a single CSV file (recommended)
python plot_ppl.py --csv logs/ppl_log.csv --smooth 7 \
                   --out figs/ppl_curve.pdf

# Example 2: If you have three separate CSVs, use path:MethodName format
python plot_ppl.py \
       --csv baseline.csv:Baseline ppo.csv:PPO a2c.csv:A2C \
       --smooth 5 \
       --out figs/ppl_curve.pdf
"""
