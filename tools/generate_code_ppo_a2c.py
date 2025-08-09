from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, torch
from tqdm import tqdm

MODEL_DIR = Path("nanoGPT-RL/saved_nanoGPT_finetuned/A2C_best_step_1600")
MBPP_FILE = "/Users/serafinayu/PycharmProjects/nanoGPT-RL/google-research/mbpp/mbpp_train.jsonl"

device   = "mps" if torch.backends.mps.is_available() else (
           "cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token      # silence warning

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()

prompts, refs, gens = [], [], []
with open(MBPP_FILE) as fh:
    for line in fh:
        obj = json.loads(line)
        prompts.append(obj["prompt"])
        refs.append(obj["completion"])

for prompt in tqdm(prompts, desc="Generating"):
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=100)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    gens.append(decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip())

out_path = MODEL_DIR.with_suffix(".jsonl")   # PPO_best_step_160.jsonl, A2C_best_step_1600.jsonl
with open(out_path, "w") as fh:
    for p, r, g in zip(prompts, refs, gens):
        fh.write(json.dumps({"prompt": p, "reference_code": r, "generated_code": g}) + "\n")

print("✅  Generation finished →", out_path)
