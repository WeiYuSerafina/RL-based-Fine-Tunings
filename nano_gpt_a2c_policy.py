import os
import json
import pickle
from typing import Optional

import torch
import torch.nn as nn

from model_a2c import GPT, GPTConfig
from transformers import GPT2Tokenizer, GPT2TokenizerFast


class NanoGPTA2CPolicy(nn.Module):
    def __init__(
        self,
        model_dir: str,
        tokenizer_path: Optional[str] = None,
        batch_size: int = 4,
        force_block_size: Optional[int] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.debug = debug

        print("✅ Initialized A2C model (NanoGPTA2CPolicy)")

        # 1. Load config.json / meta.pkl / fallback
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
        else:
            # 1.1 If not, try meta.pkl
            meta_dir = tokenizer_path or model_dir
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
                    dropout=0.0,
                )
                print(
                    f"⚠️  config.json missing, using meta.pkl from {meta_path} + default hparams"
                )
            else:
                print(
                    f"⚠️  meta.pkl also missing at {meta_path} → fallback to tokenizer defaults"
                )
                # Load the tokenizer (tok_dir is now available)
                tok_dir = tokenizer_path if tokenizer_path is not None else "./data/mbpp_new"
                self.tokenizer = GPT2TokenizerFast.from_pretrained(
                    tok_dir, local_files_only=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"✅ Tokenizer loaded from {tok_dir}")
                # Constructing a default GPTConfig with tokenizer.vocab_size
                cfg_dict = dict(
                    vocab_size=self.tokenizer.vocab_size,
                    block_size=256,
                    n_layer=4,
                    n_head=4,
                    n_embd=256,
                    bias=True,
                    dropout=0.0,
                )

        gpt_cfg = GPTConfig(**cfg_dict)

        # 2. block_size handling
        if force_block_size is not None:
            if self.debug and force_block_size != gpt_cfg.block_size:
                print(
                    f"🔧 [block_size override] {gpt_cfg.block_size} -> {force_block_size}"
                )
            gpt_cfg.block_size = force_block_size

        # 3. Initialize the GPT model
        self.model = GPT(gpt_cfg)
        n_embd = gpt_cfg.n_embd

        # 4. If block_size does not match the model default, safe extension position embedding
        if hasattr(self.model.transformer, "wpe"):
            old_wpe = self.model.transformer.wpe
            if old_wpe.num_embeddings != gpt_cfg.block_size:
                new_wpe = nn.Embedding(gpt_cfg.block_size, n_embd)
                n_copy = min(old_wpe.num_embeddings, gpt_cfg.block_size)
                with torch.no_grad():
                    new_wpe.weight[:n_copy] = old_wpe.weight[:n_copy]
                self.model.transformer.wpe = new_wpe
                if self.debug:
                    print(
                        f"Resized pos-emb: {old_wpe.num_embeddings} -> {gpt_cfg.block_size}"
                    )

        # 5. Load checkpoint  (ckpt.pt / checkpoint.pt / pytorch_model.bin)
        ckpt_path = os.path.join(model_dir, "ckpt.pt")
        if not os.path.exists(ckpt_path):
            for alt in ("checkpoint.pt", "pytorch_model.bin"):
                alt_path = os.path.join(model_dir, alt)
                if os.path.exists(alt_path):
                    ckpt_path = alt_path
                    break
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")

        state_dict = torch.load(ckpt_path, map_location="cpu")
        clean_state_dict = {}
        for key, val in state_dict.items():
            new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
            clean_state_dict[new_key] = val

        missing, unexpected = self.model.load_state_dict(clean_state_dict, strict=False)
        if self.debug:
            print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        print(f"✅ Successfully loaded A2C nanoGPT-RL model from {ckpt_path}")

        # 6. Load tokenizer
        if not hasattr(self, "tokenizer"):
            if tokenizer_path and os.path.exists(tokenizer_path):
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    tokenizer_path, local_files_only=True
                )
                print(f"✅ Successfully loaded tokenizer from {tokenizer_path}")
            elif os.path.exists(os.path.join(model_dir, "tokenizer.json")):
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model_dir, local_files_only=True
                )
                print(f"✅ Successfully loaded tokenizer from {model_dir}")
            else:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                print(
                    "⚠️ Warning: Tokenizer not found in local model dir. Using GPT2 tokenizer instead."
                )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Set pad_token to eos_token for padding support.")

        # Expose pad_token_id (trainer may depend on it)
        self.pad_token_id = self.tokenizer.pad_token_id

        # 7. Align vocab/embedding size (critical: otherwise high <unk> ratio → PPL explosion)
        self._maybe_resize_token_embeddings(self.tokenizer.vocab_size)

        # 8. A2C value head
        self.value_head = nn.Linear(n_embd, 1)

        self.device = next(self.parameters()).device

    # Helper: vocab resize
    def _maybe_resize_token_embeddings(self, new_vocab: int):
        """If model token embedding size != tokenizer vocab, resize safely."""
        old_emb = self.model.transformer.wte
        if old_emb.num_embeddings == new_vocab:
            return

        if self.debug:
            print(
                f"⚠️ vocab mismatch: model={old_emb.num_embeddings}, tokenizer={new_vocab} → resizing."
            )

        new_emb = nn.Embedding(new_vocab, old_emb.embedding_dim)
        n_copy = min(old_emb.num_embeddings, new_vocab)
        with torch.no_grad():
            new_emb.weight[:n_copy] = old_emb.weight[:n_copy]
        self.model.transformer.wte = new_emb

        # lm_head synchronization (if present)
        if hasattr(self.model, "lm_head"):
            lm = self.model.lm_head
            if isinstance(lm, nn.Linear) and lm.out_features != new_vocab:
                new_lm = nn.Linear(lm.in_features, new_vocab, bias=False)
                with torch.no_grad():
                    n_copy = min(lm.out_features, new_vocab)
                    new_lm.weight[:n_copy] = lm.weight[:n_copy]
                self.model.lm_head = new_lm

        if hasattr(self.model, "config"):
            try:
                self.model.config.vocab_size = new_vocab
            except Exception:
                pass

        if hasattr(self.model, "tie_weights"):
            try:
                self.model.tie_weights()
            except Exception:
                if self.debug:
                    print("⚠️ tie_weights() failed; continuing without tying.")

        if self.debug:
            print("🔧 token embeddings resized.")

    # Forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,  # ★ FIX: 保留可选输出隐藏层
    ):
        """
        Returns:
            logits: [B, T, V]
            value: [B, 1] (returned during training even if labels!=None; trainer requires a baseline)
            loss / lm_loss: Available during training (CrossEntropy, shift)
            hidden_states: Optional
        """
        self.device = input_ids.device

        # Calling the GPT
        try:
            logits, loss_model, hidden_states = self.model(
                idx=input_ids,
                targets=labels,
                attention_mask=attention_mask,
                return_hidden_states=True,
            )
        except TypeError:
            logits, loss_model = self.model(idx=input_ids, targets=labels)
            hidden_states = None

        # Baseline value
        if hidden_states is not None:
            last_hidden = hidden_states[-1]  # [B, T, C]
        else:
            if self.debug:
                print("⚠️  hidden_states=None → using zero baseline (degrade).")
            B = input_ids.size(0)
            device = input_ids.device
            value = torch.zeros(B, 1, device=device)
            lm_loss = self._compute_lm_loss_safe(logits, labels) if labels is not None else None
            out = {"logits": logits, "value": value}
            if lm_loss is not None:
                out["loss"] = lm_loss
                out["lm_loss"] = lm_loss
            if return_hidden_states:
                out["hidden_states"] = hidden_states
            return out

        last_token_hidden = last_hidden[:, -1, :]  # [B, C]
        value = self.value_head(last_token_hidden)  # [B, 1]

        # LM loss
        lm_loss = None
        if labels is not None:
            lm_loss = self._compute_lm_loss_safe(logits, labels)

        # Output
        out = {"logits": logits, "value": value}
        if lm_loss is not None:
            out["loss"] = lm_loss
            out["lm_loss"] = lm_loss
        if return_hidden_states:
            out["hidden_states"] = hidden_states
        return out

    # CrossEntropy calculation (shift + ignore_index=-100) + dtype safety
    @staticmethod
    def _compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    @classmethod
    def _compute_lm_loss_safe(cls, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.dtype != torch.long:
            labels = labels.long()
        if labels.size(1) != logits.size(1):
            T = min(labels.size(1), logits.size(1))
            labels = labels[:, :T]
            logits = logits[:, :T, :]
        return cls._compute_lm_loss(logits, labels)
