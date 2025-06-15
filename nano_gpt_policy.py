import torch
import torch.nn as nn
import json
import os
from model import GPT, GPTConfig
from transformers import GPT2Tokenizer
from dataclasses import fields

class NanoGPTPolicy(nn.Module):
    def __init__(self, model_dir):
        super(NanoGPTPolicy, self).__init__()

        # 1. Load config.json
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Only keep the fields accepted by GPTConfig
        valid_keys = {f.name for f in fields(GPTConfig)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}

        # Initialization
        gpt_config = GPTConfig(**filtered_config)

        # 2. Update block_size before initializing model
        gpt_config.block_size = 256

        # 3. Initialize the model
        self.model = GPT(gpt_config)
        self.config = gpt_config

        # 4. Replace position embedding to match new block size
        n_embd = gpt_config.n_embd
        self.model.transformer.wpe = torch.nn.Embedding(gpt_config.block_size, n_embd)

        # 5. Load model weights and bias
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
            new_state_dict[new_key] = state_dict[key]
        self.model.load_state_dict(new_state_dict, strict=False)  # allow new wpe shape

        print(f"Successfully loaded nanoGPT-RL model from {model_dir}")

        # 6. Load tokenizer(The baseline model and its tokenizer located differently)
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("data/arcade_new", local_files_only=True)
            print(f"Successfully loaded tokenizer from data/arcade_new")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from data/arcade_new. Error: {e}")

        # Set pad_token if it's missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token to eos_token for padding support.")

        # 7. Move model to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate(self, input_ids, max_new_tokens=100):
        block_size = self.model.config.block_size
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            context_input = generated[:, -block_size:] if generated.size(1) > block_size else generated

            output = self.forward(context_input)
            logits = output[0] if isinstance(output, tuple) else output  # [batch, seq_len, vocab]
            next_token_logits = logits[:, -1, :]

            # ✅ NaN / Inf 检查
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                print("❌ NaN or Inf detected in logits! Skipping generation.")
                break

            # ✅ 屏蔽 <|endoftext|>
            if self.tokenizer.eos_token_id is not None:
                next_token_logits[:, self.tokenizer.eos_token_id] = float('-inf')

            # 使用贪婪策略
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)

            # ✅ Word-level 重复检测（解码后）
            decoded_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            words = decoded_text.strip().split()
            from collections import Counter
            word_counts = Counter(words)
            if any(count >= 5 for count in word_counts.values()):
                print("⚠️ Detected repeated word pattern. Breaking early.")
                break

           # ✅ Token-level 重复检测（解码前）
            if generated.size(1) >= 2 and torch.equal(generated[0, -1:], generated[0, -2:-1]):
               print("⚠️ Detected token repetition. Breaking early.")
               break

        # ✅ 生成完成后，重新 forward 整句，获取 full log_probs
        with torch.no_grad():
            final_logits = self.forward(generated)
            if isinstance(final_logits, tuple):
                final_logits = final_logits[0]

            log_probs = torch.nn.functional.log_softmax(final_logits, dim=-1)

            # 获取每个生成 token 的 log_prob（不包括 prompt）
            prompt_len = input_ids.shape[-1]
            gen_token_ids = generated[:, prompt_len:]

            # ✅ 防御越界 1：prompt_len 超出 logits 长度
            if prompt_len >= log_probs.shape[1]:
                print("❌ Prompt length exceeds logits range.")
                return generated, torch.tensor(float('nan')).to(generated.device)

            # ✅ 防御越界 2：生成 token 太长
            if gen_token_ids.shape[1] > log_probs.shape[1]:
                print("❌ Generated tokens exceed logits range.")
                return generated, torch.tensor(float("nan")).to(generated.device)

            # 收集每个 token 的 log_prob
            token_log_probs = log_probs[:, prompt_len : prompt_len + gen_token_ids.shape[1], :]
            gen_log_probs = token_log_probs.gather(2, gen_token_ids.unsqueeze(-1)).squeeze(-1)

            # ✅ 返回前再次检查 log_probs 是否有效
            if torch.isnan(gen_log_probs).any() or torch.isinf(gen_log_probs).any():
                print("❌ gen_log_probs contains NaN/Inf, returning NaN tensor.")
                return generated, torch.full_like(gen_log_probs, float('nan'))

            # ✅ 返回每个 token 的 log_probs（可以求和 / 平均）
            return generated, gen_log_probs # gen_log_probs: [batch, gen_len]

    # Compatible with HuggingFace style and Karpathy-style
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(idx=input_ids, targets=labels, attention_mask=attention_mask)

    def get_log_probs(self, input_ids, labels):
        logits = self.forward(input_ids)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
