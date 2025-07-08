import os
import time
import torch
from model import GPT, GPTConfig  # 确保和你的项目结构一致

# === 设置参数 ===
model_path = "./out/mbpp_baseline_v2"  # 如果 ckpt.pt 在当前目录
checkpoint_file = os.path.join(model_path, "ckpt.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_new_tokens = 100  # 控制生成长度，可自由调整

# === 加载 checkpoint 和模型 ===
print("Loading model checkpoint...")
ckpt = torch.load(checkpoint_file, map_location=device)

model_args = ckpt['model_args']         # 包含 n_layer, n_head, n_embd 等结构参数
state_dict = ckpt['model']              # 仅模型权重
config = GPTConfig(**model_args)
model = GPT(config)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# === 构造输入 ===
# 示例 prompt：[1, 2, 3, 4] 仅供测试，可替换为真实 prompt
input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long).to(device)

# === 计时并推理 ===
print(f"Running inference for {max_new_tokens} new tokens...")
start = time.time()
output = model.generate(input_ids, max_new_tokens=max_new_tokens)
end = time.time()

# === 统计速度 ===
generated_tokens = output.shape[1] - input_ids.shape[1]
elapsed_time = end - start
inference_speed = generated_tokens / elapsed_time

# === 输出结果 ===
print(f"Generated {generated_tokens} tokens in {elapsed_time:.2f} seconds.")
print(f"Inference Speed: {inference_speed:.2f} tokens/s")
