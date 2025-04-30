import torch
import torch.nn as nn
import json
import os
from model_a2c import GPT, GPTConfig
from transformers import GPT2Tokenizer

class NanoGPTA2CPolicy(nn.Module):
    def __init__(self, model_dir):
        super(NanoGPTA2CPolicy, self).__init__()

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
        self.model.load_state_dict(new_state_dict, strict=False)

        print(f"✅ Successfully loaded A2C nanoGPT-RL model from {model_dir}")

        # 6. Load tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
            print(f"✅ Successfully loaded tokenizer from {model_dir}")
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            print("⚠️ Warning: Tokenizer not found in local model dir. Using GPT2 tokenizer instead.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Set pad_token to eos_token for padding support.")

        # 7. Add value head for A2C
        self.value_head = nn.Linear(n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        logits, _, hidden_states = self.model(input_ids, attention_mask=attention_mask, return_hidden_states=True)

        # 取最后一层的 hidden state
        last_hidden = hidden_states[-1]  # shape: [B, T, D]
        last_token_hidden = last_hidden[:, -1, :]  # shape: [B, D]

        value = self.value_head(last_token_hidden)  # shape: [B, 1]
        return logits, value

