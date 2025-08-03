import os
import time
import torch
from model import GPT, GPTConfig  # 确保和你的项目结构一致

# === 设置参数 ===
model_path = "./saved_nanoGPT_finetuned/A2C_best_step_1600"  # 模型目录
checkpoint_file = os.path.join(model_path, "ckpt.pt")       # 优先加载ckpt.pt
hf_checkpoint_file = os.path.join(model_path, "pytorch_model.bin")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_new_tokens = 100  # 控制生成长度

def remap_state_dict(sd):
    """去掉前缀并丢弃 value_head.*（推理用不到）。"""
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        if k.startswith("value_head."):
            continue
        out[k] = v
    return out

# === 加载 checkpoint 和模型 ===
print("Loading model checkpoint...")
if os.path.exists(checkpoint_file):
    # === 兼容自定义 ckpt.pt ===
    ckpt = torch.load(checkpoint_file, map_location=device)
    model_args = ckpt['model_args']         # 包含 n_layer, n_head, n_embd 等结构参数
    state_dict = ckpt['model']              # 模型权重（通常带有 model. 前缀）
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = remap_state_dict(state_dict)
    # 用 strict=False 避免 lm_head.bias 等缺失报错
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:   print("Missing keys:", missing)
    if unexpected: print("Unexpected keys:", unexpected)

elif os.path.exists(hf_checkpoint_file):
    # === 兼容 Hugging Face 风格 pytorch_model.bin ===
    state_dict = torch.load(hf_checkpoint_file, map_location=device)
    # 你需要提供 config.json 或自己定义参数（保持你原先写法，不加新功能）
    model_args = {
        "vocab_size": 50257,
        "block_size": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256
    }
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = remap_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:   print("Missing keys:", missing)
    if unexpected: print("Unexpected keys:", unexpected)

else:
    raise FileNotFoundError("No ckpt.pt or pytorch_model.bin found in model_path!")

model.to(device)
model.eval()

# === 构造输入 ===
input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long).to(device)

# === 计时并推理 ===
print(f"Running inference for {max_new_tokens} new tokens...")
start = time.time()
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
end = time.time()

# === 统计速度 ===
generated_tokens = output.shape[1] - input_ids.shape[1]
elapsed_time = end - start
inference_speed = generated_tokens / elapsed_time

# === 输出结果 ===
print(f"Generated {generated_tokens} tokens in {elapsed_time:.2f} seconds.")
print(f"Inference Speed: {inference_speed:.2f} tokens/s")
