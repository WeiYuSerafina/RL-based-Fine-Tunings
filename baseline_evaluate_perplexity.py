# baseline_evaluate_perplextiy.py

import os
import json
import math
import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from model import GPT, GPTConfig  # 你的 baseline nanoGPT 模型定义

def load_baseline_model(
    model_path: str,
    tokenizer_path: str,
    device: torch.device,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    block_size: int = 256,
):
    """
    加载 baseline GPT 模型与 tokenizer。
    """
    # tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    # 模型
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config).to(device).eval()
    # 自动选择 ckpt
    if os.path.isdir(model_path):
        ckpt = os.path.join(model_path, "ckpt_step900.pt")
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(model_path, "ckpt.pt")
    else:
        ckpt = model_path
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get("model_state_dict", sd.get("model", sd))
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model, tokenizer

def load_prompt_completion_pairs(path: str, max_samples: int = 1000):
    """
    和 PPO/A2C 脚本一致：读取 jsonl or json，提取 (prompt, prompt+completion) 对
    """
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    else:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    pairs = []
    for obj in records:
        prompt = obj.get("prompt", "").strip()
        completion = (obj.get("completion") or obj.get("code", "")).strip()
        if prompt and completion:
            pairs.append((prompt, f"{prompt} {completion}"))
            if len(pairs) >= max_samples:
                break
    return pairs

def evaluate_baseline_perplexity(
    model: GPT,
    tokenizer: GPT2TokenizerFast,
    prompt_full_pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 256,
):
    """
    按批处理，但每个样本内部屏蔽 prompt，只评估 completion 部分：
      - labels 前 prompt_len 置 -100
      - loss.sum * valid_tokens / total_tokens -> exp -> PPL
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token; please set one before padding.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 常用做法
    tokenizer.padding_side = "right"

    for i in range(0, len(prompt_full_pairs), batch_size):
        batch = prompt_full_pairs[i : i + batch_size]
        prompts, fulls = zip(*batch)

        enc = tokenizer(
            list(fulls),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)

        # 构造 labels：屏蔽 prompt + padding
        labels = input_ids.clone()
        for j, prompt in enumerate(prompts):
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            plen = min(len(prompt_ids), labels.size(1))
            labels[j, :plen] = -100 # 忽略 prompt
        labels[attn_mask == 0] = -100  # 忽略 padding

        with torch.no_grad():
            # 位置参数调用：idx=input_ids, targets=labels, attention_mask=None
            out = model(idx=input_ids, targets=labels)
            # GPT.forward 返回 (logits, loss)
            if isinstance(out, tuple) and len(out) == 2:
                _, loss = out
            else:
                raise RuntimeError(f"Unexpected model output: {out}")

        # 累积 NLL
        n_valid = (labels != -100).sum().item()
        total_nll += loss.item() * n_valid
        total_tokens += n_valid

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_baseline_model(
        args.model_path, args.tokenizer_path, device
    )
    pairs = load_prompt_completion_pairs(args.jsonl_path, max_samples=args.max_samples)
    print(f"📊 Evaluating baseline on {len(pairs)} samples…")
    ppl = evaluate_baseline_perplexity(
        model, tokenizer, pairs, device,
        batch_size=args.batch_size, max_length=args.max_length
    )
    print(f"✅ Baseline Perplexity: {ppl:.2f}")

"""
python3 baseline_evaluate_perplexity.py \
  --model_path ./out/mbpp_baseline_v3/ \
  --tokenizer_path ./data/mbpp_new \
  --jsonl_path ./google-research/mbpp/sanitized-mbpp.json \
  --max_samples 500 \
  --batch_size 8 \
  --max_length 256
"""