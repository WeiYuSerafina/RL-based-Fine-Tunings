import json, difflib, csv
import numpy as np
import matplotlib.pyplot as plt

# 1) Configure your file paths
# Define the model names and their corresponding result files
files = {
    "Baseline": "./baseline_generated_results.jsonl",
    "PPO":      "./ppo_best_step_160_generated_results.jsonl",
    "A2C":      "./a2c_best_step_1600_generated_results.jsonl",
}

# Compute string similarity between generated code and reference
def code_similarity(gen, ref):
    return difflib.SequenceMatcher(None, gen.strip(), ref.strip()).ratio()

# Load all similarity values from a JSONL file
def load_sims(path):
    sims = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            sims.append(code_similarity(obj["generated_code"], obj["reference_code"]))
    return np.array(sims, dtype=float)

# 2) Compute mean/std and print to console
# Iterate over models, compute similarity statistics
results = []
for name, path in files.items():
    sims = load_sims(path)
    mean = float(np.mean(sims))
    std  = float(np.std(sims))
    results.append((name, mean, std, len(sims)))
    print(f"{name}: mean={mean:.4f}, std={std:.4f}, n={len(sims)}")

# 3) Save results to CSV (for tables or further analysis)
# Write model similarity summary to CSV
with open("code_similarity_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "MeanSimilarity", "StdDev", "NumSamples"])
    w.writerows(results)

# 4) Plot bar chart (for main paper)
# Plot average similarity for each model
labels = [r[0] for r in results]
means  = [r[1] for r in results]

plt.figure(figsize=(6, 3), dpi=100)
plt.bar(labels, means)
plt.ylabel("Average String Similarity")
plt.title("Code Correctness (Approximated by String Similarity)")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("code_correctness_similarity.png", dpi=300)
plt.close()

# 5) Optional: Plot similarity distribution (for appendix)
# Create box plot to show per-sample similarity distribution
all_sims = [load_sims(files[k]) for k in labels]
plt.figure(figsize=(6, 3), dpi=100)
plt.boxplot(all_sims, labels=labels, showmeans=True)
plt.ylabel("String Similarity")
plt.title("Distribution of Code Similarity per Sample")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("code_similarity_boxplot.png", dpi=300)
plt.close()

print("Saved: code_similarity_summary.csv, code_correctness_similarity.png, code_similarity_boxplot.png")
