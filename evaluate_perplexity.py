import os
import json
import math
import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from model import GPT, GPTConfig  # 你 nanoGPT 的自定义模型

# === 配置 ===
block_size = 256
stride = 128
device = torch.device("cpu")
model_path = "./saved_nanoGPT"
tokenizer_path = "data/arcade_new"
jsonl_path = "arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v2.jsonl"

# === 加载 tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# === 初始化模型配置（与你训练时保持一致） ===
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=2,
    n_head=2,
    n_embd=128
)
model = GPT(config).to(device)

# === 加载模型参数 ===
state_dict = torch.load(os.path.join(model_path, "model.pt"), map_location=device)
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

print("✅ 模型权重加载成功，共参数量：", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# === 准备 tokenized 序列和 masked labels ===
input_ids_all = []
label_ids_all = []

with open(jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        prompt = data.get("prompt", "").strip()
        completion = data.get("completion", "").replace("<|endoftext|>", "").satrip()

        if prompt and completion:
            full_input = prompt + "\n" + completion
            full_enc = tokenizer(full_input, return_tensors="pt")
            prompt_enc = tokenizer(prompt, return_tensors="pt")

            input_ids = full_enc.input_ids[0]
            labels = input_ids.clone()

            # Mask prompt部分的label（不计算其loss）
            labels[: prompt_enc.input_ids.shape[1]] = -100

            input_ids_all.append(input_ids)
            label_ids_all.append(labels)

# === 合并所有样本 ===
input_ids_cat = torch.cat(input_ids_all)
labels_cat = torch.cat(label_ids_all)
seq_len = input_ids_cat.size(0)

print(f"✅ 拼接后总 token 数量: {seq_len}")

# === 计算 perplexity（滑动窗口）===
nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + block_size, seq_len)
    trg_len = end_loc - prev_end_loc
    input_chunk = input_ids_cat[begin_loc:end_loc].unsqueeze(0).to(device)
    label_chunk = labels_cat[begin_loc:end_loc].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_chunk)
        logits = logits[:, :-1, :].contiguous()
        labels_shifted = label_chunk[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels_shifted.view(-1))

        nll_sum += loss.sum().item()
        n_tokens += (labels_shifted != -100).sum().item()

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

# === 计算并输出 perplexity ===
if n_tokens == 0:
    print("❌ 没有有效 token 被用于 perplexity 计算。请检查数据。")
    exit()

avg_nll = nll_sum / n_tokens
ppl = math.exp(avg_nll)

print("\n✅ 评估完成：")
print(f"Average NLL: {avg_nll:.4f}")
print(f"Perplexity (only on completion): {ppl:.2f}")
