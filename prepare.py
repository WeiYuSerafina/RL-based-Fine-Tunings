import os
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer

# === CONFIGURATION ===
dataset_path = 'arcade-nl2code/arcade_nl2code/annotated_dataset/new_tasks_for_nanoGPT.jsonl'  # your input .jsonl file
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
    for line in tqdm(lines, desc=f"Encoding {split} set"):
        item = json.loads(line)
        prompt = item["prompt"]
        completion = item["completion"]
        full_text = prompt + "\n" + completion + tokenizer.eos_token
        token_ids = tokenizer.encode(full_text)
        ids.extend(token_ids)
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

