import os
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer

# === CONFIGURATION ===
dataset_path = 'arcade-nl2code/arcade_nl2code/annotated_dataset/converted_new_tasks.jsonl'  # your input .jsonl file
output_dir = 'data/arcade_new'                      # output folder
val_ratio = 0.1                                 # 10% validation split
model_name = 'gpt2'                             # tokenizer type

# === LOAD TOKENIZER ===
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # in case it's needed

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
    skipped = 0  # 记录跳过样本数

    for line in tqdm(lines, desc=f"Encoding {split} set"):
        item = json.loads(line)
        instruction = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        solution = item.get("solution", "").strip()

        if not instruction or not solution:
            skipped += 1
            continue

        full_text = f"Instruction: {instruction}\nContext: {context}\n{solution}"

        # ✅ 跳过只包含 <|endoftext|> 或太短的样本
        if full_text.strip() == tokenizer.eos_token or len(full_text.strip()) <= 20:
            skipped += 1
            continue

        token_ids = tokenizer.encode(full_text)
        ids.extend(token_ids)

    print(f"⚠️ Skipped {skipped} invalid or short samples in {split} set.")
    return ids

# === ENCODE ===
train_ids = encode(train_lines, "train")
val_ids = encode(val_lines, "val")

# === SAVE BIN FILES ===
os.makedirs(output_dir, exist_ok=True)

train_bin_path = os.path.join(output_dir, 'train.bin')
val_bin_path = os.path.join(output_dir, 'val.bin')

np.array(train_ids, dtype=np.uint16).tofile(train_bin_path)
np.array(val_ids, dtype=np.uint16).tofile(val_bin_path)

# === SAVE META INFO ===
meta = {
    'tokenizer': model_name,
    'train_size': len(train_ids),
    'val_size': len(val_ids),
    'vocab_size': tokenizer.vocab_size,
}

with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Done! Saved train/val bin files and meta info to: {output_dir}/")

print("Train token count:", len(train_ids))
print("Val token count:", len(val_ids))

# === SAVE tokenizer ===
tokenizer.save_pretrained(output_dir)
print("Tokenizer saved to:", output_dir)

# === preview_training_samples.py ===
import numpy as np
from transformers import GPT2Tokenizer

def preview_training_samples(bin_path, tokenizer_path, n=5, token_limit=20000):
    # 加载 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path) # ("data/arcade_new")
    tokenizer.pad_token = tokenizer.eos_token

    # 加载 token 二进制文件
    train_data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    decoded_text = tokenizer.decode(train_data[:token_limit])
    samples = decoded_text.split(tokenizer.eos_token)

    print(f"\n📦 Decoding {n} full training samples:")
    for i, sample in enumerate(samples[:n]):
        print(f"\n📌 Sample {i + 1}:\n{sample.strip()}")

if __name__ == "__main__":
    # 用你的训练路径替换这里
    preview_training_samples(
        bin_path="data/arcade_new/train.bin",
        tokenizer_path="data/arcade_new",
        n=5,
        token_limit=20000
    )
