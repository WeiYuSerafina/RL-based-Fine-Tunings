import os
import json
import math
import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from model import GPT, GPTConfig  # baseline nanoGPT 模型

# Configuration
block_size = 256
stride = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./out/mbpp_baseline_v3/" # "./out/mbpp_baseline_v3/"
tokenizer_path = "data/mbpp_new" # data/mbpp_new
json_path = "/Users/serafinayu/PycharmProjects/nanoGPT-RL/google-research/mbpp/sanitized-mbpp.json" #sanitized-mbpp.json

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# Initialize and load the Baseline model
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=256
)
model = GPT(config).to(device)

# Automatically select checkpoints
if os.path.isdir(model_path):
    ckpt_file = os.path.join(model_path, "ckpt_step900.pt")
    if not os.path.isfile(ckpt_file):
        ckpt_file = os.path.join(model_path, "ckpt.pt")
else:
    ckpt_file = model_path

print(f"Loading baseline checkpoint from {ckpt_file}...")
ckpt = torch.load(ckpt_file, map_location=device)

# Support different storage formats
state_dict = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt

# Clean prefix
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Loaded baseline model, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Read data and build token sequence
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

# Merge
input_ids_cat = torch.cat(input_ids_all)
labels_cat = torch.cat(label_ids_all)
seq_len = input_ids_cat.size(0)
print(f"Total tokens: {seq_len}")

# Calculate PPL
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
