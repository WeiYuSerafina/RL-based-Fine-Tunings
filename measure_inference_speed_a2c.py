import os
import time
import torch
from model import GPT, GPTConfig

# Configuration
model_path = "./saved_nanoGPT_finetuned/A2C_best_step_1600"
checkpoint_file = os.path.join(model_path, "ckpt.pt")
hf_checkpoint_file = os.path.join(model_path, "pytorch_model.bin")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_new_tokens = 100

def remap_state_dict(sd):
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

# Load checkpoint and model
print("Loading model checkpoint...")
if os.path.exists(checkpoint_file):
    # Compatible with custom ckpt.pt
    ckpt = torch.load(checkpoint_file, map_location=device)
    model_args = ckpt['model_args']
    state_dict = ckpt['model']
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = remap_state_dict(state_dict)
    # Use strict=False to avoid missing errors such as lm_head.bias
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:   print("Missing keys:", missing)
    if unexpected: print("Unexpected keys:", unexpected)

elif os.path.exists(hf_checkpoint_file):
    # Compatible with Hugging Face style pytorch_model.bin
    state_dict = torch.load(hf_checkpoint_file, map_location=device)
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

# Construct Input
input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long).to(device)

# Time and inference
print(f"Running inference for {max_new_tokens} new tokens...")
start = time.time()
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
end = time.time()

# Speed
generated_tokens = output.shape[1] - input_ids.shape[1]
elapsed_time = end - start
inference_speed = generated_tokens / elapsed_time

# Result
print(f"Generated {generated_tokens} tokens in {elapsed_time:.2f} seconds.")
print(f"Inference Speed: {inference_speed:.2f} tokens/s")
