from transformers import GPT2Tokenizer
import json
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

lengths = []

with open("google-research/mbpp/mbpp_train.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)

        prompt = item.get("prompt", "").strip()
        completion = item.get("completion", "").strip()
        full_text = f"{prompt}\n{completion}"

        tokens = tokenizer.encode(full_text)
        lengths.append(len(tokens))

print(f"Total number of samples: {len(lengths)}")
print(f"Maximum token length: {np.max(lengths)}")
print(f"Average token length: {np.mean(lengths):.2f}")
print(f"Median token length: {np.median(lengths)}")
print(f"90% of samples have fewer than: {np.percentile(lengths, 90):.0f} tokens")
