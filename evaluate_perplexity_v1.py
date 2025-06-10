import os
import json
import math
from tqdm import tqdm
import torch
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

from model import GPT, GPTConfig  # 使用 nanoGPT 的模型定义

# === 配置路径 ===
model_path = "./saved_nanoGPT"
tokenizer_path = "data/arcade_new"
jsonl_path = "arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v1.jsonl"
device = torch.device("cpu")
block_size = 256
stride = 128

# === 加载 tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# === 初始化并加载 nanoGPT 模型 ===
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=2,
    n_head=2,
    n_embd=128)  # 注意修改成你训练时用的配置

# === 初始化模型 ===
model = GPT(config).to(device)

# === 加载 state_dict ===
state_dict = torch.load(os.path.join(model_path, "model.pt"), map_location=device)

# === 去掉 _orig_mod. 前缀（如果存在）===
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# === 加载权重 ===
model.load_state_dict(state_dict)
model.eval()

# === 打印参数数量确认 ===
print("✅ Model weights loaded successfully, total parameters：", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# === 提取所有 code_context 字段中的代码 ===
code_snippets = []
with open(jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if "completion" in data and data["completion"].strip():
            cleaned_code = data["completion"].replace("<|endoftext|>", "").strip()
            code_snippets.append(cleaned_code)

if not code_snippets:
    print("❌ No completion field was extracted, please check the jsonl file structure.")
    exit()

# === 合并所有代码段作为模型输入 ===
joined_text = "\n\n".join(code_snippets)
encodings = tokenizer(joined_text, return_tensors="pt")
input_ids = encodings.input_ids[0]

seq_len = input_ids.size(0)
nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0

print(f"✅ Withdraw token quantity: {seq_len}，Start calculating perplexity...")

# === 滑动窗口评估 perplexity ===
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + block_size, seq_len)
    trg_len = end_loc - prev_end_loc
    input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)
    target_chunk = input_chunk.clone()
    target_chunk[:, :-trg_len] = -100  # 只评估当前段的后 trg_len 个 token

    # === 自定义 nanoGPT，不支持 labels 参数, 改为自己计算交叉熵 loss ===
    with torch.no_grad():
        # === 模型前向传播 ===
        logits = model(input_chunk)  # shape: [1, T, vocab_size]

        # === Shift logits and targets to align for next-token prediction ===
        logits = logits[:, :-1, :].contiguous()  # [1, T-1, vocab_size]
        targets = target_chunk[:, 1:].contiguous()  # [1, T-1]

        # === 交叉熵损失 ===
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        neg_log_likelihood = loss.mean()

    nll_sum += neg_log_likelihood.item() * trg_len
    n_tokens += trg_len
    prev_end_loc = end_loc

    if end_loc == seq_len:
        break

# === 输出最终 Perplexity ===
avg_nll = nll_sum / n_tokens
ppl = math.exp(avg_nll)

print("\n✅ Assessment Completed：")
print(f"Average NLL: {avg_nll:.4f}")
print(f"Perplexity: {ppl:.2f}")
