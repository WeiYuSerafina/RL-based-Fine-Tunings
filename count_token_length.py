from transformers import GPT2Tokenizer
import json
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 防止 pad 报错

lengths = []

with open("arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v1.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)

        prompt = item.get("prompt", "").strip() # 已封装Instruction, Context, Preamble, Step-by-Step，且已拼接好
        completion = item.get("completion", "").strip() # 只包含目标代码（或解决方案）+ `<
        full_text = f"{prompt}\n{completion}"

        tokens = tokenizer.encode(full_text)
        lengths.append(len(tokens))

# 输出 token 分布信息
print(f"Total number of samples: {len(lengths)}")
print(f"Maximum token length: {np.max(lengths)}")
print(f"Average token length: {np.mean(lengths):.2f}")
print(f"Median token length: {np.median(lengths)}")
print(f"90% of samples have fewer than: {np.percentile(lengths, 90):.0f} tokens")
