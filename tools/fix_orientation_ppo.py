import json, torch
from pathlib import Path
from transformers import GPT2Config, GPT2LMHeadModel, logging

logging.set_verbosity_error()            # Disable transformers warnings

MODEL_DIR = Path("nanoGPT-RL/saved_nanoGPT_finetuned/PPO_best_step_160")
BIN_FILE  = MODEL_DIR / "pytorch_model.bin"
CFG_FILE  = MODEL_DIR / "config.json"

# 1) Load and complete the config.json if needed
cfg = json.load(open(CFG_FILE))
cfg.setdefault("model_type", "gpt2")
blk = cfg.get("block_size") or cfg.get("n_positions") or 256
cfg["n_positions"] = cfg["n_ctx"] = blk
json.dump(cfg, open(CFG_FILE, "w"), indent=2)

# 2) Build an empty model on CPU using the updated config
g_cfg = GPT2Config.from_json_file(CFG_FILE)
model = GPT2LMHeadModel(g_cfg)

# 3) Load raw weights and transpose (out, in) matrices if needed
state = torch.load(BIN_FILE, map_location="cpu")
ref   = model.state_dict()
for k, v in state.items():
    if k in ref and v.ndim == 2 and v.shape[::-1] == ref[k].shape:
        state[k] = v.t()
model.load_state_dict(state, strict=False)

# 4) Save the model in Hugging Face format (overwrite original files)
model.save_pretrained(MODEL_DIR, safe_serialization=False)
print("✅ Transposition completed and model saved to →", MODEL_DIR)
