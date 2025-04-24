import torch
import torch.nn as nn
import json
import os
from model import GPT, GPTConfig
from transformers import GPT2Tokenizer

class NanoGPTPolicy(nn.Module):
    def __init__(self, model_dir):
        super(NanoGPTPolicy, self).__init__()

        # 1. Load config.json
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        gpt_config = GPTConfig(**config_dict)

        # 2. Update block_size before initializing model
        gpt_config.block_size = 128

        # 3. Initialize the model
        self.model = GPT(gpt_config)

        # 4. Replace position embedding to match new block size
        n_embd = gpt_config.n_embd
        self.model.transformer.wpe = torch.nn.Embedding(gpt_config.block_size, n_embd)

        # 5. Load model weights
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
            new_state_dict[new_key] = state_dict[key]
        self.model.load_state_dict(new_state_dict, strict=False)  # ⚠️ allow new wpe shape

        print(f"✅ Successfully loaded nanoGPT-RL model from {model_dir}")

        # 6. Load tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
            print(f"✅ Successfully loaded tokenizer from {model_dir}")
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            print("⚠️ Warning: Tokenizer not found in local model dir. Using GPT2 tokenizer instead.")

        # Set pad_token if it's missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Set pad_token to eos_token for padding support.")

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)[0]

    def generate(self, input_ids, max_new_tokens=100):
        block_size = self.model.config.block_size
        generated = input_ids.clone()  # 防止修改原始 input_ids
        all_log_probs = []

        for _ in range(max_new_tokens):
            # 剪裁上下文以适配 block_size
            context_input = generated[:, -block_size:] if generated.size(1) > block_size else generated

            output = self.forward(context_input)
            logits = output[0] if isinstance(output, tuple) else output  # [batch, seq_len, vocab]
            next_token_logits = logits  # 只取最后一个位置 [batch, vocab]

            # Sampling with temperature
            temperature = 0.8
            probs = torch.softmax(next_token_logits / temperature, dim=-1)

            # 防止模型重复预测同一个 token（可选加 top-k）
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # 修复 batch size 不一致问题
            if next_token.size(0) != generated.size(0):
                next_token = next_token[:generated.size(0), :]

            # 计算 log prob
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            max_log_prob = torch.gather(log_probs, dim=1, index=next_token).squeeze(-1)
            all_log_probs.append(max_log_prob)

            # 拼接
            generated = torch.cat((generated, next_token), dim=1)

            # ---- 防止模型陷入循环复读 ----
            decoded_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            if "ructionruction" in decoded_text or decoded_text.count("Instruction") > 1:
                print("Detected repetition artifact. Breaking early.")
                break

        all_log_probs = torch.stack(all_log_probs, dim=1)
        return generated, all_log_probs

    def get_log_probs(self, input_ids, labels):
        logits = self.forward(input_ids)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
