import torch
import torch.nn.functional as F

class PPOTrainer:
    # === 1. Constructor function: Initialize PPOTrainer ===
    def __init__(self, model, tokenizer, optimizer, buffer, clip_epsilon=0.2, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.buffer = buffer
        self.clip_epsilon = clip_epsilon
        self.config = config

        # 设置设备（GPU 优先）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 检查logits正常性
        self._check_logits_sanity()

        # 初始化 NaN 计数器
        self.nan_counter = 0

        self.max_grad_norm = 0.5  # GPT-2 微调（RLHF / PPO-0.25, 0.5, 最多 1.0

        self.update_step = 0

        # ------- compatible with self.model.config --------
        if not hasattr(self.model, "config") and hasattr(self.model, "model"):
            self.model.config = self.model.model.config

        # === 2. Model sanity check: ensure that the logits under dummy input have no NaN and the mean is normal ===
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


    # === 3. Generate output for a batch of prompts using the current strategy for subsequent reward evaluation ===
    def rollout(self, prompts, max_len=None):
        """
        使用当前策略对多个 prompts 做 rollout，返回：
        - generated_texts：生成的文本（用于 reward 计算）
        - prompts：原始输入
        - generated_ids_list：生成的 token ids（调试用）
        - prompt_lens：每个 prompt 的 token 长度（用于后续 log_probs 对齐）
        """
        if max_len is None:
            max_len = getattr(self.config, "max_new_tokens", 100)  # 默认值为 100

        self.model.eval()
        generated_texts = []
        generated_ids_list = []
        prompt_lens = []

        for prompt in prompts:
            # 编码 prompt 为 input_ids
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            prompt_len = input_ids.shape[-1]  # 记录当前 prompt 的长度
            prompt_lens.append(prompt_len)  # 添加到列表中

            with torch.no_grad():
                # 使用贪婪策略生成 token，不返回 log_probs（后续整句再 forward）
                generated_ids, _ = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_len
                )

            # 保存生成的 token ids
            generated_ids_list.append(generated_ids)

            # 解码生成结果（去掉 prompt 部分）
            new_token_ids = generated_ids[0, input_ids.shape[-1]:]
            generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts, prompts, generated_ids_list, prompt_lens


    # === 4. Calculate PPO loss function (Clipped Objective + KL penalty) ===
    def ppo_loss(self, old_log_probs, new_log_probs, rewards):
        """
        Using Advantage instead of Reward to calculate PPO losses
        """
        # 1. 计算比例项 ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        # PPO Clip Range: clip_epsilon sets 0.2 as default
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # 2. 检查 reward 是否异常
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("❌ Invalid rewards detected (NaN or Inf), skipping loss computation.")
            return torch.tensor(float('nan')).to(rewards.device)

        # 3. 计算 Advantage（代替 reward），此处用 reward - mean 作为近似 Advantage
        # Advantages 择标准化：
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages = torch.clamp(advantages, -5.0, 5.0)

        # 注意：如果没有 value function，这种 "centered reward" 是合理的近似
        # advantages = rewards - rewards.mean()

        # 4. PPO核心损失计算（用 Advantage 替代 reward）
        ppo_core_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        #5. KL 惩罚项（使用动态系数）
        # 可选：传入 step 或 epoch 参数
        step = getattr(self, "update_step", 0)  # 若未设定 step，默认为 0
        # 动态 KL 系数（早期较小，后期增强以防策略发散）
        kl_coeff = 0.1 if step < 200 else 0.3
        # 计算 KL 差异并加权
        kl_div = torch.mean(old_log_probs - new_log_probs)
        kl_penalty = kl_coeff * kl_div

        # 6.总损失
        loss =  ppo_core_loss  + 0.1 * kl_penalty

        # 7. loss 数值异常检查
        if torch.isnan(loss):
            print("❌ NaN loss detected in PPO. Dumping debug info:")
            print("old_log_probs:", old_log_probs)
            print("new_log_probs:", new_log_probs)
            print("ratio:", ratio)
            print("clipped_ratio:", clipped_ratio)
            print("rewards:", rewards)
            print("loss (raw):", loss)
            return torch.tensor(float('nan')).to(loss.device)

        # 8. loss 为负值（理论上不应该）的特殊处理（可选）
        if loss.item() < 0:
            print(f"⚠️ Warning: PPO loss is negative ({loss.item():.4f}), check reward or log_prob stability.")

        self._last_policy_loss = ppo_core_loss.detach()
        self._last_kl_penalty = kl_penalty.detach()

        return loss


    # === 5. 拼接 prompt + generated，forward 后提取生成部分的 token log_probs，并按句子维度累加 ===
    def compute_log_probs(self, inputs, actions, prompt_lens):
        """
        对拼接后的 [prompt + generated] 整句 forward，再提取 generated 部分的 log_probs。
        """
        input_texts = list(inputs)
        action_texts = list(actions)

        # 拼接 prompt + generated → 用于模型 forward
        joined_texts = [p.strip() + " " + a.strip() for p, a in zip(input_texts, action_texts)]

        # 编码整句（prompt + generated）
        joined_encodings = self.tokenizer(
            joined_texts,
            return_tensors="pt",
            # 如果 prompt 长度不一致，tokenizer 会对较短的输入右侧填充 <pad> token
            padding=True,
            truncation=True,
            max_length=self.model.config.block_size
        ).to(self.model.device)

        input_ids = joined_encodings.input_ids
        # attention_mask 会确保模型只处理有效 token（非 padding）
        attention_mask = joined_encodings.attention_mask

        # 编码 generated（用于获取 generated token 长度）
        action_encodings = self.tokenizer(
            action_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.model.config.block_size
        ).to(self.model.device)

        gen_lengths = action_encodings.attention_mask.sum(dim=1)  # 每条 generated 的长度 [B]

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

        # softmax 得到 log_probs
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

        # 获取生成部分的 token 序列（prompt + generated → 截取最后 gen_length 个 token）
        gathered_log_probs = []
        for i in range(input_ids.size(0)):
            gen_len = gen_lengths[i]
            seq_len = input_ids[i].shape[0]
            if gen_len == 0 or gen_len > seq_len:
                gathered_log_probs.append(torch.tensor(float('nan')).to(self.model.device))
                continue

            # （⚠️ 注意：可能还需要 min(prompt_len + gen_len, seq_len) 截断保护）
            prompt_len = prompt_lens[i]
            gen_token_ids = input_ids[i][prompt_len:prompt_len + gen_len]
            gen_log_probs = log_probs[i][prompt_len:prompt_len + gen_len, :]

            # gen_token_ids = input_ids[i][-gen_len:]  # [gen_len]
            # gen_log_probs = log_probs[i][-gen_len:, :]  # [gen_len, V]

            # 检查是否越界（token_id > vocab_size）
            vocab_size = log_probs.shape[-1]
            if (gen_token_ids >= vocab_size).any():
                print(f"❌ Token ID 超出 vocab 范围，跳过该样本")
                gathered_log_probs.append(torch.tensor(float('nan')).to(self.model.device))
                continue

            # 安全 gather
            selected = gen_log_probs.gather(1, gen_token_ids.unsqueeze(1)).squeeze(1)  # [gen_len]
            gathered_log_probs.append(selected.sum())

        final_log_probs = torch.stack(gathered_log_probs)  # [B]

        if torch.isnan(final_log_probs).any() or torch.isinf(final_log_probs).any():
            print("❌ final_log_probs 含非法值")
            return torch.full((final_log_probs.size(0),), float('nan')).to(self.model.device)

        return final_log_probs


    # === 6. 从 buffer 中采样，标准化 reward，重新计算 log_probs，执行 PPO 损失计算与优化更新；包含大量错误检查与 debug 输出 ===
    def update(self, buffer, batch_size=None):
        """
        PPO 更新函数（每次从 buffer 采样数据，执行更新）
        """
        actual_batch_size = len(buffer) if batch_size is None else min(batch_size, len(buffer))

        if actual_batch_size == 0:
            print("⚠️ PPO skipped: not enough data in buffer.")
            return

        prompts, generated_codes, rewards, old_log_probs = buffer.sample(actual_batch_size)

        # 🛡️ 检查数据合法性（防止空 list）
        if not prompts or not generated_codes:
            print("⚠️ Empty prompts or completions, skipping update.")
            return

        # ✅ [新增] reward tensor 转换 + 极小值防御
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)

        # ✅ 标准化 rewards，并加极小值防御（防止 std 太小导致爆炸）
        rewards_std = rewards.std()
        if rewards_std < 1e-8:
            print("⚠️ reward.std() is too small; skipping standardization")
        else:
            rewards = (rewards - rewards.mean()) / (rewards_std + 1e-8)

        # ✅ 添加 reward clamp 限制范围，防止极端值爆炸
        rewards = torch.clamp(rewards, -5.0, 5.0)
        print(f"📊 Reward range: min={rewards.min():.4f}, max={rewards.max():.4f}")


        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("❌ Invalid reward in buffer (NaN or Inf), skipping update.")
            return

        if rewards.abs().max() < 1e-3:
            print(f"⚠️ Reward values too small: {rewards.tolist()} → skipping PPO step.")
            return


        # ✅ 1. 检查拼接后长度是否超过 block_size（防止 forward 崩溃）
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

        # ✅ 2. 检查是否存在 None 或空字符串输入
        for i, (prompt, gen) in enumerate(zip(prompts, generated_codes)):
            if prompt is None or gen is None or prompt.strip() == "" or gen.strip() == "":
                print(f"⚠️ Sample {i} has invalid input: prompt or generated is empty or None.")
                return

        # ✅ 3. 打印关键输入，定位崩溃用
        def safe_print(label, text, max_len=1000):
            text = text.replace("\n", " ")
            if len(text) > max_len:
                text = text[:max_len] + " ...[truncated]"
            print(f"{label} {text}")
        safe_print("🧪 Prompt example:", prompt)
        safe_print("🧪 Generated example:", gen)
        input_ids = self.tokenizer(prompts[0] + generated_codes[0], return_tensors="pt").input_ids
        print("🧪 Full input len:", input_ids.size(1))

        # 重新计算新策略下的 log_probs（现在是整句）
        # 计算当前策略对已生成代码generated_codes 的 log probability（对数概率），以便用于 PPO 策略更新
        prompt_lens = [len(self.tokenizer(p, return_tensors='pt').input_ids[0]) for p in prompts]
        new_log_probs = self.compute_log_probs(prompts, generated_codes, prompt_lens)

        # NaN/Inf 检查
        if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
            print("❌ NaN or Inf detected in new_log_probs, skipping this update.")
            return

        # old_log_probs 检查
        if torch.isnan(old_log_probs).any() or torch.isinf(old_log_probs).any():
            print("❌ Invalid old_log_probs in buffer, skipping this update.")
            return

        # reward tensor 检查
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("❌ Invalid rewards (NaN/Inf), skipping PPO loss.")
            return

        # 对齐尺寸防护
        if new_log_probs.size() != old_log_probs.size():
            print(f"❌ Size mismatch: new_log_probs {new_log_probs.size()}, old_log_probs {old_log_probs.size()}")
            return

        # 计算 PPO 损失
        loss = self.ppo_loss(old_log_probs, new_log_probs, rewards)

        # 检查 loss 是否崩溃
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ NaN or Inf detected in PPO loss, skipping this update.")
            self.nan_counter += 1
            if self.nan_counter >= 3:
                print("❌ Exiting training due to 3 consecutive NaN losses.")
                exit()  # 或 raise RuntimeError("Too many NaNs in PPO")
            return
        else:
            self.nan_counter = 0  # 恢复正常时重置计数器

        # 反向传播 + 参数更新/Backpropagation + parameter update
        self.optimizer.zero_grad()
        loss.backward()

        # 监控梯度范数/Monitoring the gradient norm
        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None
        ]))
        print(f"🚨 Pre-clip Gradient Norm: {total_norm.item():.4f}")

        # Gradient clipping: Gradient explosion fix, limit the maximum gradient norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.optimizer.step()

        # ✅ 更新完成后 +1
        self.update_step += 1

        print(f"✂️ Applied gradient clipping with max_norm = {self.max_grad_norm}")

        print(f"✅ PPO Update Done! Loss = {loss.item():.4f}")

        self.last_stats = {
            "total_loss": loss.item(),
            "policy_loss": self._last_policy_loss.item(),
            "kl_penalty": self._last_kl_penalty.item(),
        }




