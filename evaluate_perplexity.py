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
model_path = "out/mbpp_baseline_v2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer_path = "data/mbpp_new"
jsonl_path = "/Users/serafinayu/PycharmProjects/nanoGPT-RL/google-research/mbpp/mbpp_train.jsonl"

# === 加载 tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# === 初始化模型配置（与你训练时保持一致） ===
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=256
)
model = GPT(config).to(device)
"""
# === 加载模型参数 ===
state_dict = torch.load(os.path.join(model_path, "model.pt"), map_location=device)
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

print("✅ 模型权重加载成功，共参数量：", sum(p.numel() for p in model.parameters()) / 1e6, "M")
"""

# === 加载 checkpoint ===
checkpoint_file = os.path.join(model_path, "ckpt.pt")  # 或 "checkpoint.pt" 如果文件叫这个
ckpt = torch.load(checkpoint_file, map_location=device)
state_dict = ckpt['model']  # 提取模型权重部分

# === 清洗可能存在的 _orig_mod. 前缀 ===
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# === 加载权重到模型 ===
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
        completion = data.get("completion", "").replace("<|endoftext|>", "").strip()

        if prompt and completion:
            # full_input = prompt + "\n" + completion
            # full_enc = tokenizer(full_input, return_tensors="pt")
            # prompt_enc = tokenizer(prompt, return_tensors="pt")
            # 用 <|endoftext|> 作为分隔符
            full_input = prompt + tokenizer.eos_token + completion
            # 计算 prompt 部分长度时，要把这个 eos_token 一并算进去
            prompt_enc = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt")
            # 再整体编码
            full_enc = tokenizer(full_input, return_tensors="pt")

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

print(f"✅ The total number of tokens after concatenation: {seq_len}")

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
    print("❌ No valid token was used in the perplexity calculation. Please check the dataset.")
    exit()

avg_nll = nll_sum / n_tokens
ppl = math.exp(avg_nll)

print("\n✅ Evaluation completed：")
print(f"Average NLL: {avg_nll:.4f}")
print(f"Perplexity (only on completion): {ppl:.2f}")
