import os
import json
import random
import pickle
import numpy as np
import re
from tqdm import tqdm
from transformers import GPT2Tokenizer

# === CONFIGURATION ===
dataset_path = 'arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v1.jsonl'
output_dir = 'data/arcade_new'
val_ratio = 0.1
model_name = 'gpt2'
block_size = 256

# === LOAD TOKENIZER ===
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Save tokenizer to output_dir which vocab.json, merges.txt, tokenizer_config.json will be saved
tokenizer.save_pretrained(output_dir)

"""
def is_too_long(prompt, completion):
    total = tokenizer(prompt + completion)["input_ids"]
    return len(total) > block_size
"""

# === READ JSONL DATA ===
print(f"Reading data from {dataset_path}...")
with open(dataset_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.shuffle(lines)
n_total = len(lines)
n_val = int(val_ratio * n_total)
train_lines = lines[:-n_val]
val_lines = lines[-n_val:]

def encode(lines, split):
    ids = []
    skipped = 0

    for line in tqdm(lines, desc=f"Encoding {split} set"):
        try:
            item = json.loads(line)
            prompt = item.get("prompt", "").strip()
            completion = item.get("completion", "").strip()
            if not prompt or not completion:
                skipped += 1
                continue

            full_text = f"{prompt}\n{completion}"

            token_ids = tokenizer.encode(full_text)

            if len(token_ids) > block_size:
                skipped += 1
                continue

            ids.extend(token_ids)
        except Exception:
            skipped += 1
            continue

    print(f"⚠️ Skipped {skipped} samples in {split} set.")
    return ids

# === ENCODE ===
train_ids = encode(train_lines, "train")
val_ids = encode(val_lines, "val")

# === SAVE BIN FILES ===
os.makedirs(output_dir, exist_ok=True)
np.array(train_ids, dtype=np.uint16).tofile(os.path.join(output_dir, 'train.bin'))
np.array(val_ids, dtype=np.uint16).tofile(os.path.join(output_dir, 'val.bin'))

# === SAVE META ===
meta = {
    'tokenizer': model_name,
    'vocab_size': tokenizer.vocab_size,
    'train_size': len(train_ids),
    'val_size': len(val_ids),
    'block_size': block_size,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved tokenized train/val to: {output_dir}/")
print("Train token count:", len(train_ids))
print("Val token count:", len(val_ids))

# === PREVIEW ===
def preview_training_samples(bin_path, tokenizer_path, n=10, token_limit=20000):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    train_data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    decoded_text = tokenizer.decode(train_data[:token_limit])
    samples = decoded_text.split(tokenizer.eos_token)
    print(f"\n📦 Preview {n} training samples:")
    for i, sample in enumerate(samples[:n]):
        print(f"\n📌 Sample {i + 1}:\n{sample.strip()}")

if __name__ == "__main__":
    preview_training_samples(
        bin_path="data/arcade_new/train.bin",
        tokenizer_path="data/arcade_new",
        n=10,
        token_limit=20000
    )
