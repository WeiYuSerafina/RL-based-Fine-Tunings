import torch
import torch.nn.functional as F

class PPOTrainer:
    # 1. Constructor function: Initialize PPOTrainer
    def __init__(self, model, tokenizer, optimizer, buffer, clip_epsilon=0.2, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.buffer = buffer
        self.clip_epsilon = clip_epsilon
        self.config = config

        # Set device (GPU first)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Check logits normality
        self._check_logits_sanity()

        # Initialize the NaN counter
        self.nan_counter = 0
        self.max_grad_norm = 0.5
        self.update_step = 0

        # compatible with self.model.config
        if not hasattr(self.model, "config") and hasattr(self.model, "model"):
            self.model.config = self.model.model.config

    # 2. Model sanity check: ensure that the logits under dummy input have no NaN and the mean is normal
    def _check_logits_sanity(self):
        self.model.eval()
        with torch.no_grad():
            dummy_input = self.tokenizer("print('Hello')", return_tensors="pt").input_ids.to(self.device)
            output = self.model(dummy_input)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            mean = logits.float().mean().item()
            print("✅ Dummy logits mean:", mean)
            assert not torch.isnan(logits).any(), "❌ NaN found in dummy logits!"
            assert abs(mean) < 100, "⚠️ Logits mean unusually large! Possible instability."

    # 3. Generate output for a batch of prompts using the current strategy for subsequent reward evaluation
    def rollout(self, prompts, max_len=None):
        """
        Rollout multiple prompts using the current policy, returning:
        - generated_texts: Generated text (used for reward calculation)
        - prompts: Original input
        - generated_ids_list: Generated token IDs (for debugging)
        - prompt_lens: Token length of each prompt (used for subsequent log_probs alignment)
        """
        if max_len is None:
            max_len = getattr(self.config, "max_new_tokens", 100)  # The default value is 100

        self.model.eval()
        generated_texts = []
        generated_ids_list = []
        prompt_lens = []

        for prompt in prompts:
            # Encode prompt as input_ids
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            prompt_len = input_ids.shape[-1]  # Record the length of the current prompt
            prompt_lens.append(prompt_len)  # Add to List

            with torch.no_grad():
                # Generate tokens using a greedy policy and do not return log_probs
                generated_ids, _ = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_len
                )

            # Save the generated token ids
            generated_ids_list.append(generated_ids)

            # Decode the generated result (remove the prompt part)
            new_token_ids = generated_ids[0, input_ids.shape[-1]:]
            generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts, prompts, generated_ids_list, prompt_lens

    # 4. Calculate PPO loss function (Clipped Objective + KL penalty)
    def ppo_loss(self, old_log_probs, new_log_probs, rewards):
        """
        Using Advantage instead of Reward to calculate PPO losses
        """
        # 1). Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        # PPO Clip Range: clip_epsilon sets 0.2 as default
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # 2). Check whether the reward is abnormal
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("❌ Invalid rewards detected (NaN or Inf), skipping loss computation.")
            return torch.tensor(float('nan')).to(rewards.device)

        # 3). Calculate Advantage (instead of reward), here use reward - mean as the approximate Advantage
        # Advantages of Standardization:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages = torch.clamp(advantages, -5.0, 5.0)

        # Note: If there is no value function, this "centered reward" is a reasonable approximation
        # advantages = rewards - rewards.mean()

        # 4). PPO core loss calculation (using Advantage instead of reward)
        ppo_core_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 5). KL penalty (using dynamic coefficients)
        # Optional: Pass in step or epoch parameters
        step = getattr(self, "update_step", 0)  # If step is not set, the default is 0
        # Dynamic KL coefficient (small in the early stages, increased later to prevent strategy divergence)
        kl_coeff = 0.1 if step < 200 else 0.3
        # Calculate KL divergence and weight it
        kl_div = torch.mean(old_log_probs - new_log_probs)
        kl_penalty = kl_coeff * kl_div

        # 6).Total loss
        loss =  ppo_core_loss  + 0.1 * kl_penalty

        # 7). Loss value abnormality check
        if torch.isnan(loss):
            print("❌ NaN loss detected in PPO. Dumping debug info:")
            print("old_log_probs:", old_log_probs)
            print("new_log_probs:", new_log_probs)
            print("ratio:", ratio)
            print("clipped_ratio:", clipped_ratio)
            print("rewards:", rewards)
            print("loss (raw):", loss)
            return torch.tensor(float('nan')).to(loss.device)

        # 8). Special handling for negative loss values (which theoretically shouldn't be the case) (optional)
        if loss.item() < 0:
            print(f"⚠️ Warning: PPO loss is negative ({loss.item():.4f}), check reward or log_prob stability.")

        self._last_policy_loss = ppo_core_loss.detach()
        self._last_kl_penalty = kl_penalty.detach()

        return loss

    # 5. Concatenate prompt + generated, forward, and extract the generated token log_probs, and accumulate them by sentence dimension
    def compute_log_probs(self, inputs, actions, prompt_lens):
        """
        Forward the concatenated sentence [prompt + generated] and extract the log_probs of the generated part.
        """
        input_texts = list(inputs)
        action_texts = list(actions)

        # Splice prompt + generated → for model forward
        joined_texts = [p.strip() + " " + a.strip() for p, a in zip(input_texts, action_texts)]

        # Encode the entire sentence (prompt + generated)
        joined_encodings = self.tokenizer(
            joined_texts,
            return_tensors="pt",
            # If the prompt lengths are inconsistent, the tokenizer will right-pad the shorter input with a <pad> token.
            padding=True,
            truncation=True,
            max_length=self.model.config.block_size
        ).to(self.model.device)

        input_ids = joined_encodings.input_ids
        # The attention_mask ensures that the model only processes valid tokens (non-padding)
        attention_mask = joined_encodings.attention_mask

        # Encoding generated (used to obtain the length of generated token)
        action_encodings = self.tokenizer(
            action_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.model.config.block_size
        ).to(self.model.device)

        gen_lengths = action_encodings.attention_mask.sum(dim=1)

        try:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]
        except Exception as e:
            print(f"❌ model.forward exception: {e}")
            return torch.full((input_ids.size(0),), float('nan')).to(self.model.device)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("❌ logits contains NaN or Inf, skip log_prob calculation")
            return torch.full((logits.size(0),), float('nan')).to(logits.device)

        # softmax to get log_probs
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

        # Get the generated token sequence (prompt + generated → intercept the last gen_length tokens)
        gathered_log_probs = []
        for i in range(input_ids.size(0)):
            gen_len = gen_lengths[i]
            seq_len = input_ids[i].shape[0]
            if gen_len == 0 or gen_len > seq_len:
                gathered_log_probs.append(torch.tensor(float('nan')).to(self.model.device))
                continue

            prompt_len = prompt_lens[i]
            gen_token_ids = input_ids[i][prompt_len:prompt_len + gen_len]
            gen_log_probs = log_probs[i][prompt_len:prompt_len + gen_len, :]

            # gen_token_ids = input_ids[i][-gen_len:]  # [gen_len]
            # gen_log_probs = log_probs[i][-gen_len:, :]  # [gen_len, V]

            # Check if it is out of bounds (token_id > vocab_size)
            vocab_size = log_probs.shape[-1]
            if (gen_token_ids >= vocab_size).any():
                print(f"❌ Token ID 超出 vocab 范围，跳过该样本")
                gathered_log_probs.append(torch.tensor(float('nan')).to(self.model.device))
                continue

            # Security gather
            selected = gen_log_probs.gather(1, gen_token_ids.unsqueeze(1)).squeeze(1)  # [gen_len]
            gathered_log_probs.append(selected.sum())

        final_log_probs = torch.stack(gathered_log_probs)  # [B]

        if torch.isnan(final_log_probs).any() or torch.isinf(final_log_probs).any():
            print("❌ final_log_probs 含非法值")
            return torch.full((final_log_probs.size(0),), float('nan')).to(self.model.device)

        return final_log_probs

    # 6. Sample from the buffer, normalize the reward, recalculate log_probs, perform PPO loss calculation and optimization update; contains extensive error checking and debug output
    def update(self, buffer, batch_size=None):
        """
        PPO update function (sample data from the buffer each time and perform updates)
        """
        actual_batch_size = len(buffer) if batch_size is None else min(batch_size, len(buffer))

        if actual_batch_size == 0:
            print("⚠️ PPO skipped: not enough data in buffer.")
            return

        prompts, generated_codes, rewards, old_log_probs = buffer.sample(actual_batch_size)

        # Check data validity (prevent empty list)
        if not prompts or not generated_codes:
            print("⚠️ Empty prompts or completions, skipping update.")
            return

        # Reward tensor conversion + minimum value defense
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)

        # Normalize rewards and add minimum value protection (to prevent std from being too small and causing explosion)
        rewards_std = rewards.std()
        if rewards_std < 1e-8:
            print("⚠️ reward.std() is too small; skipping standardization")
        else:
            rewards = (rewards - rewards.mean()) / (rewards_std + 1e-8)

        # Add reward clamp to limit the range to prevent extreme value explosion
        rewards = torch.clamp(rewards, -5.0, 5.0)
        print(f"Reward range: min={rewards.min():.4f}, max={rewards.max():.4f}")

        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("❌ Invalid reward in buffer (NaN or Inf), skipping update.")
            return

        if rewards.abs().max() < 1e-3:
            print(f"⚠️ Reward values too small: {rewards.tolist()} → skipping PPO step.")
            return

        # Check if the concatenated length exceeds block_size (to prevent forward from crashing)
        for i, (prompt, gen) in enumerate(zip(prompts, generated_codes)):
            try:
                input_ids = self.tokenizer(prompt + gen, return_tensors="pt").input_ids
                if input_ids.size(1) > self.model.config.block_size:
                    print(
                        f"⚠️ Sample {i} skipped: length {input_ids.size(1)} > block_size {self.model.config.block_size}")
                    return
            except Exception as e:
                print(f"❌ Tokenization error at sample {i}: {e}")
                return

        # Checks for the presence of None or an empty string input
        for i, (prompt, gen) in enumerate(zip(prompts, generated_codes)):
            if prompt is None or gen is None or prompt.strip() == "" or gen.strip() == "":
                print(f"⚠️ Sample {i} has invalid input: prompt or generated is empty or None.")
                return

        # Print key input to locate crashes
        def safe_print(label, text, max_len=1000):
            text = text.replace("\n", " ")
            if len(text) > max_len:
                text = text[:max_len] + " ...[truncated]"
            print(f"{label} {text}")
        safe_print("🧪 Prompt example:", prompt)
        safe_print("🧪 Generated example:", gen)
        input_ids = self.tokenizer(prompts[0] + generated_codes[0], return_tensors="pt").input_ids
        print("🧪 Full input len:", input_ids.size(1))

        # Recalculate log_probs under the new policy (now a complete sentence)
        # Calculate the log probability of the current policy for the generated codes generated_codes for use in PPO policy updates
        prompt_lens = [len(self.tokenizer(p, return_tensors='pt').input_ids[0]) for p in prompts]
        new_log_probs = self.compute_log_probs(prompts, generated_codes, prompt_lens)

        # NaN/Inf checks
        if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
            print("❌ NaN or Inf detected in new_log_probs, skipping this update.")
            return

        # old_log_probs checks
        if torch.isnan(old_log_probs).any() or torch.isinf(old_log_probs).any():
            print("❌ Invalid old_log_probs in buffer, skipping this update.")
            return

        # reward tensor checks
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("❌ Invalid rewards (NaN/Inf), skipping PPO loss.")
            return

        # Alignment dimension protection
        if new_log_probs.size() != old_log_probs.size():
            print(f"❌ Size mismatch: new_log_probs {new_log_probs.size()}, old_log_probs {old_log_probs.size()}")
            return

        # Calculate PPO loss
        loss = self.ppo_loss(old_log_probs, new_log_probs, rewards)

        # Check if loss crashes
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ NaN or Inf detected in PPO loss, skipping this update.")
            self.nan_counter += 1
            if self.nan_counter >= 3:
                print("❌ Exiting training due to 3 consecutive NaN losses.")
                exit()  # raise RuntimeError("Too many NaNs in PPO")
            return
        else:
            self.nan_counter = 0  # Reset the counter when back to normal

        # Backpropagation + parameter update
        self.optimizer.zero_grad()
        loss.backward()

        # Monitoring the gradient norm
        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None
        ]))
        print(f"🚨 Pre-clip Gradient Norm: {total_norm.item():.4f}")

        # Gradient clipping: Gradient explosion fix, limit the maximum gradient norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.optimizer.step()

        # +1 after update completed
        self.update_step += 1

        print(f"✂️ Applied gradient clipping with max_norm = {self.max_grad_norm}")
        print(f"✅ PPO Update Done! Loss = {loss.item():.4f}")

        self.last_stats = {
            "total_loss": loss.item(),
            "policy_loss": self._last_policy_loss.item(),
            "kl_penalty": self._last_kl_penalty.item(),
        }




