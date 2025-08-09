import os, json, pickle, torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
)

MODEL_DIR = Path("nanoGPT-RL/out/mbpp_baseline_v3")
MBPP_FILE = "/Users/serafinayu/PycharmProjects/nanoGPT-RL/google-research/mbpp/mbpp_train.jsonl"

def ensure_hf_model_package(model_dir: Path):
    """Convert ckpt.pt to a Hugging Face model if config.json is missing."""
    cfg_file = model_dir / "config.json"
    if cfg_file.exists():
        return  # Already a Hugging Face model

    ckpt_file = model_dir / "ckpt.pt"
    meta_file = model_dir / "meta.pkl"
    if not ckpt_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"{model_dir} is missing config.json and no ckpt.pt or meta.pkl found for conversion."
        )

    # ---- 1) Load meta.pkl to infer model hyperparameters (nanoGPT convention) ----
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    # Support both meta and meta['config']
    conf = meta.get("config", meta)
    n_layer = int(conf.get("n_layer", 4))
    n_head  = int(conf.get("n_head", 4))
    n_embd  = int(conf.get("n_embd", 256))
    block_size = int(conf.get("block_size", conf.get("n_positions", 256)))
    vocab_size = int(conf.get("vocab_size", 50257))

    # ---- 2) Build Hugging Face-compatible model (using GPT-2 config) ----
    hf_config = GPT2Config(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        n_positions=block_size, n_ctx=block_size,
        vocab_size=vocab_size, bos_token_id=50256, eos_token_id=50256
    )
    model = GPT2LMHeadModel(hf_config)

    # ---- Load and normalize the state_dict from ckpt ----
    ckpt = torch.load(ckpt_file, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Transpose (out, in) weights to (in, out) if needed
    ref = model.state_dict()
    fixed = {}
    for k, v in state.items():
        if k in ref and v.ndim == 2:
            exp_shape = ref[k].shape
            if v.shape == (exp_shape[1], exp_shape[0]):
                v = v.t()
        fixed[k] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    print("[convert] missing:", len(missing), "unexpected:", len(unexpected))

    # ---- 4) Load tokenizer using vocab.json and merges.txt ----
    tok = GPT2TokenizerFast(vocab_file=str(model_dir / "vocab.json"),
                            merges_file=str(model_dir / "merges.txt"))
    # If special_tokens_map.json exists, load and add special tokens
    sp_map = model_dir / "special_tokens_map.json"
    if sp_map.exists():
        with open(sp_map, "r") as f:
            sp = json.load(f)
        tok.add_special_tokens({k: v for k, v in sp.items() if isinstance(v, str)})
        # Update token embeddings
        model.resize_token_embeddings(len(tok))

    # ---- 5) Save as Hugging Face model package ----
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    print(f"[convert] Saved Hugging Face package to: {model_dir}")

def main():
    ensure_hf_model_package(MODEL_DIR)

    # Choose device: prioritize MPS (Apple), then CUDA, then CPU
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model (model_dir is now a valid HF package)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.to(device).eval()

    # Load MBPP data
    prompts, reference_codes = [], []
    with open(MBPP_FILE, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj.get("prompt") or obj.get("instruction") or "")
            reference_codes.append(obj.get("completion") or obj.get("reference_code") or "")

    # Generate code
    generated_codes = []
    for prompt in tqdm(prompts, desc="Generating code"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from generated output if duplicated
        gen = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
        generated_codes.append(gen)

    # Save results
    out = f"{MODEL_DIR.name}_generated_results.jsonl"
    with open(out, "w") as f:
        for p, r, g in zip(prompts, reference_codes, generated_codes):
            f.write(json.dumps({"prompt": p, "reference_code": r, "generated_code": g}) + "\n")
    print(f"✅ Generation completed. Results saved to: {out}")

if __name__ == "__main__":
    main()
