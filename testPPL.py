import os
import json
import math
import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from model import GPT, GPTConfig  # baseline nanoGPT 模型

# === 配置 ===
block_size = 256
stride = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./saved_nanoGPT_finetuned/A2C_best_step_1400/" # "./out/mbpp_baseline_v3/"
tokenizer_path = "saved_nanoGPT_finetuned/A2C_best_step_1400" # data/mbpp_new
json_path = "/Users/serafinayu/PycharmProjects/nanoGPT-RL/google-research/mbpp/sanitized-mbpp.json" #sanitized-mbpp.json

# === 加载 tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# === 初始化并加载 Baseline 模型 ===
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=256
)
model = GPT(config).to(device)

# 先尝试 HuggingFace 保存格式
hf_bin = os.path.join(model_path, "pytorch_model.bin")
if os.path.isfile(hf_bin):
    print(f"Loading HuggingFace-style weights from {hf_bin}...")
    state_dict = torch.load(hf_bin, map_location=device)
else:  # 退回旧的 ckpt 逻辑
    ckpt_file = os.path.join(model_path, "ckpt_step900.pt")
    if not os.path.isfile(ckpt_file):
        ckpt_file = os.path.join(model_path, "ckpt.pt")
    print(f"Loading baseline checkpoint from {ckpt_file}...")
    ckpt = torch.load(ckpt_file, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
"""
# 去掉可能的 _orig_mod. 前缀
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Loaded model, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
"""
# 清洗键名
clean_sd = {}
for k, v in state_dict.items():
    k = k.removeprefix("model.")        # Python 3.9 没有 removeprefix 就用切片
    k = k.replace("_orig_mod.", "")
    if k.startswith("value_head"):
        continue
    clean_sd[k] = v

missing, unexpected = model.load_state_dict(clean_sd, strict=False)
print(f"✅ Loaded model · missing: {len(missing)} · ignored extra: {len(unexpected)}")
print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# === 读取数据并构建 token 序列 ===
input_ids_all = []
label_ids_all = []
with open(json_path, 'r', encoding='utf-8') as f:
    records = json.load(f) if json_path.endswith('.json') else [json.loads(line) for line in f]
for obj in records:
    if not isinstance(obj, dict):
        continue
    prompt = obj.get('prompt', '').strip()
    completion = (obj.get('completion') or obj.get('code', '')).replace('<|endoftext|>', '').strip()
    if not prompt or not completion:
        continue
    full = prompt + tokenizer.eos_token + completion
    enc_full = tokenizer(full, return_tensors='pt')
    enc_prompt = tokenizer(prompt + tokenizer.eos_token, return_tensors='pt')
    ids = enc_full.input_ids[0]
    labels = ids.clone()
    labels[:enc_prompt.input_ids.size(1)] = -100
    input_ids_all.append(ids)
    label_ids_all.append(labels)

# 合并
input_ids_cat = torch.cat(input_ids_all)
labels_cat = torch.cat(label_ids_all)
seq_len = input_ids_cat.size(0)
print(f"Total tokens: {seq_len}")

# === 计算 PPL ===
nll_sum = 0.0
n_tokens = 0
prev_end = 0
for start in tqdm(range(0, seq_len, stride)):
    end = min(start + block_size, seq_len)
    inp = input_ids_cat[start:end].unsqueeze(0).to(device)
    labs = labels_cat[start:end].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        logits = logits[:, :-1, :].contiguous()
        shifted = labs[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), shifted.view(-1), ignore_index=-100, reduction='none'
        )
        nll_sum += loss.sum().item()
        n_tokens += (shifted != -100).sum().item()
    if end == seq_len:
        break
avg_nll = nll_sum / n_tokens
ppl = math.exp(avg_nll)
print(f"Average NLL: {avg_nll:.4f}, Perplexity: {ppl:.2f}")
