import json
import difflib
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_prompt_similarity(jsonl_path):
    similarities = []
    hard_copy_threshold = 0.6
    hard_copy_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {jsonl_path}"):
            data = json.loads(line)
            prompt = data.get('prompt', '').strip()
            generated = data.get('generated_code', '').strip()

            similarity = difflib.SequenceMatcher(None, prompt, generated).ratio()
            similarities.append(similarity)

            if similarity > hard_copy_threshold:
                hard_copy_count += 1

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    hard_copy_rate = hard_copy_count / len(similarities) if similarities else 0.0

    return avg_similarity, hard_copy_rate

def main():
    paths = {
        'Baseline': './baseline_generated_results.jsonl',
        'PPO': './ppo_best_step_160_generated_results.jsonl',
        'A2C': './a2c_best_step_1600_generated_results.jsonl'
    }

    avg_similarities = []
    hard_copy_rates = []

    print("\n=== Prompt Repeat Penalty Statistics ===\n")
    for name, path in paths.items():
        avg_sim, hard_rate = compute_prompt_similarity(path)
        avg_similarities.append(avg_sim)
        hard_copy_rates.append(hard_rate)
        print(f"{name}:\n  Avg Prompt Similarity = {avg_sim:.4f}\n  Hard Copy Rate (>0.6) = {hard_rate:.2%}\n")

    # Avg Prompt Similarity
    plt.figure(figsize=(6, 3), dpi=100)
    plt.bar(paths.keys(), avg_similarities)
    plt.ylim(0, 1.0)
    plt.ylabel('Average Prompt Similarity')
    plt.title('Prompt Repeat Penalty – Avg Prompt Similarity')
    plt.tight_layout()
    plt.savefig('prompt_similarity_bar.png', dpi=300)
    plt.show()

    # Hard Copy Rate
    plt.figure(figsize=(6, 3), dpi=100)
    plt.bar(paths.keys(), hard_copy_rates)
    plt.ylim(0, 1.0)
    plt.ylabel('Hard Copy Rate (Similarity > 0.6)')
    plt.title('Prompt Repeat Penalty – Hard Copy Rate')
    plt.tight_layout()
    plt.savefig('hard_copy_rate_bar.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
