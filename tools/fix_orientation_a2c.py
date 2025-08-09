# Transpose only for (out, in) → (in, out) matrices
import torch, json
from pathlib import Path
from transformers import GPT2Config, GPT2LMHeadModel, logging

logging.set_verbosity_error()        # Suppress warning messages

MODEL_DIR = Path("nanoGPT-RL/saved_nanoGPT_finetuned/A2C_best_step_1600")
CFG  = MODEL_DIR / "config.json"
BIN  = MODEL_DIR / "pytorch_model.bin"

# 1) Ensure config.json has required fields
cfg = json.load(open(CFG))
cfg.setdefault("model_type", "gpt2")
blk = cfg.get("block_size") or cfg.get("n_positions") or 256
cfg["n_positions"] = cfg["n_ctx"] = blk
json.dump(cfg, open(CFG, "w"), indent=2)

# 2) Initialize empty model on CPU
model = GPT2LMHeadModel(GPT2Config.from_json_file(CFG))

# 3) Load weights and transpose matrices with mismatched dimensions
sd_raw = torch.load(BIN, map_location="cpu")
sd_fix = {}
ref = model.state_dict()
for k, v in sd_raw.items():
    if k in ref and v.ndim == 2 and v.shape[::-1] == ref[k].shape:
        v = v.t()
    sd_fix[k] = v

missing, unexpected = model.load_state_dict(sd_fix, strict=False)
print(f"Transposed {len(sd_fix)-len(unexpected)} matrices; {len(missing)} keys missing.")

# 4) Save updated model
model.save_pretrained(MODEL_DIR, safe_serialization=False)
print("✅ Transposition complete and model saved to →", MODEL_DIR)
