# -*- coding: utf-8 -*-
"""
Refactored NanoGPTPolicy (符合 ⚠️ 必改 + 💡 建议)：
  • 设备一致性：始终将张量搬到 self.device
  • tokenizer 路径与 baseline 一致
  • checkpoint 键名自动探测 + 前缀清洗
  • forward/tuple 兼容
  • 生成循环改进（避免 CPU/GPU mismatch）
"""

import os, json, pickle
from collections import Counter
from dataclasses import fields

import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast

from model import GPT, GPTConfig
from typing import Optional


class NanoGPTPolicy(nn.Module):
    def __init__(self, model_dir: str, tokenizer_path: Optional[str] = None):
        super().__init__()

        # ---------- 0. 设备 ---------- #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- 1. 读取 config.json ---------- #
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
        else:
            # ---------- 1.1 如果没有，则尝试 meta.pkl ----------
            meta_dir = tokenizer_path or model_dir  # ← 关键行
            meta_path = os.path.join(meta_dir, "meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                cfg_dict = dict(
                    vocab_size=meta["vocab_size"],
                    block_size=meta["block_size"],
                    n_layer=4,
                    n_head=4,
                    n_embd=256,
                    bias=True,
                    dropout=0.0
                )
                print(f"⚠️  config.json missing, using meta.pkl from {meta_path} + default hparams")
            else:
                print(f"⚠️  meta.pkl also missing at {meta_path} → fallback to tokenizer defaults")
                # ----- 加载 tokenizer （此时 tok_dir 已可用） -----
                tok_dir = tokenizer_path if tokenizer_path is not None else "./data/mbpp_new"
                self.tokenizer = GPT2TokenizerFast.from_pretrained(tok_dir, local_files_only=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"✅ Tokenizer loaded from {tok_dir}")
                # ----- 用 tokenizer.vocab_size 构造缺省 GPTConfig -----
                cfg_dict = dict(
                    vocab_size=self.tokenizer.vocab_size,
                    block_size=256,
                    n_layer=4, n_head=4, n_embd=256,
                    bias=True, dropout=0.0
                )

        gpt_cfg = GPTConfig(**cfg_dict)

        # ---------- 2. 初始化模型 ---------- #
        self.model = GPT(gpt_cfg)
        self.model.transformer.wpe = nn.Embedding(gpt_cfg.block_size, gpt_cfg.n_embd)
        self.model.to(self.device)

        # ---------- 3. 加载 checkpoint ---------- #
        # 建议改为——直接使用传入的 model_dir：
        ckpt_path = os.path.join(model_dir, "ckpt.pt")  # 首选 ckpt.pt
        if not os.path.exists(ckpt_path):  # 兜底找 checkpoint.pt 或 pytorch_model.bin
            for alt in ("checkpoint.pt", "pytorch_model.bin"):
                alt_path = os.path.join(model_dir, alt)
                if os.path.exists(alt_path):
                    ckpt_path = alt_path
                    break
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 自动探测键
        state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        assert isinstance(state_dict, dict), "未找到 state_dict，请检查 checkpoint"

        clean_state = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")  # torch.compile
            if k.startswith("module."):
                k = k[len("module."):]
            clean_state[k] = v
        self.model.load_state_dict(clean_state, strict=False)
        print(f"✅ Baseline weights loaded from {ckpt_path}")

        # ---------- 4. 加载 tokenizer ---------- #
        tok_dir = tokenizer_path or "./data/mbpp_new"
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tok_dir, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ Tokenizer loaded from {tok_dir}")

    # ------------------------------------------------------------
    # 生成 + 安全检查
    # ------------------------------------------------------------
    def generate(self, input_ids, max_new_tokens: int = 100):
        input_ids = input_ids.to(self.device)
        generated = input_ids.clone()
        block_size = self.model.config.block_size

        for step in range(max_new_tokens):
            context = generated[:, -block_size:] if generated.size(1) > block_size else generated
            # -------- 兼容三种返回格式 --------
            out = self.forward(context)
            if isinstance(out, tuple):  # (logits, loss)
                logits = out[0]
            elif hasattr(out, "logits"):  # transformers.ModelOutput
                logits = out.logits
            else:  # 已是张量
                logits = out
            # ----------------------------------
            temperature = 0.9 # 0.8-1.2
            top_k = 60

            # 取最后一步 logits 并缩放温度
            next_token_logits = logits[:, -1, :] / temperature

            # NaN / Inf 检查
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                print("❌ NaN/Inf in logits, abort generation")
                break

            # 屏蔽 <|endoftext|>
            if self.tokenizer.eos_token_id is not None:
                next_token_logits[:, self.tokenizer.eos_token_id] = float("-inf")

            # ---- top-k 截断 ----
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")

            # --- softmax 取概率 & 采样  ---
            pros = torch.softmax(next_token_logits, dim=-1)  # 概率分布（float32）
            next_token = torch.multinomial(pros, num_samples=1) # 随机采样 1 个 token
            generated = torch.cat([generated, next_token], dim=1)

            # token-level 连续重复检测（每步）
            if generated.size(1) > 1 and torch.equal(generated[0, -1:], generated[0, -2:-1]):
                print("⚠️ Token repetition, break")
                break

            # word-level 重复检测（每 8 步解码一次，阈值放宽到 10）
            if step % 8 == 0:  # 每 8 步再 decode 一次即可
                decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                counts = Counter(decoded.split())
                if any(c >= 10 for c in counts.values()):
                    break

        # ---- 返回 log_probs ----
        with torch.no_grad():
            out = self.forward(generated)
            # -------- 再次三分支取 logits --------
            if isinstance(out, tuple):
                logits = out[0]
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            # ------------------------------------
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            prompt_len = input_ids.size(1)
            gen_ids = generated[:, prompt_len:]

            if prompt_len >= log_probs.size(1) or gen_ids.size(1) > log_probs.size(1):
                return generated, torch.tensor(float("nan"), device=self.device)

            token_lp = log_probs[:, prompt_len:prompt_len + gen_ids.size(1), :]
            gen_lp = token_lp.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
            if torch.isnan(gen_lp).any() or torch.isinf(gen_lp).any():
                return generated, torch.full_like(gen_lp, float("nan"))
            return generated, gen_lp

    # ------------------------------------------------------------
    # forward 兼容 Karpathy + HF
    # ------------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.model(idx=input_ids, targets=labels, attention_mask=attention_mask)
        return out  # 可能是 logits 或 (logits, loss)

    def get_log_probs(self, input_ids, labels):
        logits = self.forward(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        lp = torch.nn.functional.log_softmax(logits, dim=-1)
        return lp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
