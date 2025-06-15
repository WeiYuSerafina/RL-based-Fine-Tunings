import torch
import torch.nn.functional as F

class PPOTrainer:
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

    def rollout(self, prompts, max_len=None):
        """
        使用当前策略对多个 prompts 做 rollout，返回：
        - 生成的文本 generated_texts（用于计算 reward）
        - prompts（原始输入，方便与生成对齐）
        - generated_ids_list（生成的 token ids，用于调试或进一步分析）
        """
        if max_len is None:
            max_len = getattr(self.config, "max_new_tokens", 100)  # ✅ 默认值为 100

        self.model.eval()
        generated_texts = []
        generated_ids_list = []

        for prompt in prompts:
            # 编码 prompt 为 input_ids
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

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

        return generated_texts, prompts, generated_ids_list


    def ppo_loss(self, old_log_probs, new_log_probs, rewards):
        """
        计算 PPO 损失函数
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # 1. 检查 reward 是否异常
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("❌ Invalid rewards detected (NaN or Inf), skipping loss computation.")
            return torch.tensor(float('nan')).to(rewards.device)

        # 2. PPO核心损失计算
        loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()

        # 3. loss 数值异常检查
        if torch.isnan(loss):
            print("❌ NaN loss detected in PPO. Dumping debug info:")
            print("🔍 old_log_probs:", old_log_probs)
            print("🔍 new_log_probs:", new_log_probs)
            print("🎯 ratio:", ratio)
            print("🎯 clipped_ratio:", clipped_ratio)
            print("🎯 rewards:", rewards)
            print("🎯 loss (raw):", loss)
            return torch.tensor(float('nan')).to(loss.device)

        # ✅ 4. loss 为负值（理论上不应该）的特殊处理（可选）
        if loss.item() < 0:
            print(f"⚠️ Warning: PPO loss is negative ({loss.item():.4f}), check reward or log_prob stability.")

        return loss

    def compute_log_probs(self, inputs, actions):
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
            padding=True,
            truncation=True,
            max_length=self.model.model.config.block_size
        ).to(self.model.device)

        input_ids = joined_encodings.input_ids
        attention_mask = joined_encodings.attention_mask

        # 编码 generated（用于获取 generated token 长度）
        action_encodings = self.tokenizer(
            action_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.model.config.block_size
        ).to(self.model.device)

        action_ids = action_encodings.input_ids
        gen_lengths = action_encodings.attention_mask.sum(dim=1)  # 每条 generated 的长度 [B]

        try:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]
        except Exception as e:
            print(f"❌ model.forward 异常: {e}")
            return torch.full((input_ids.size(0),), float('nan')).to(self.model.device)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("❌ logits 中含 NaN 或 Inf，跳过 log_prob 计算")
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

            gen_token_ids = input_ids[i][-gen_len:]  # [gen_len]
            gen_log_probs = log_probs[i][-gen_len:, :]  # [gen_len, V]

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
        print("🧪 Prompt example:", prompts[0][:80].replace('\n', ' '))
        print("🧪 Generated example:", generated_codes[0][:80].replace('\n', ' '))
        input_ids = self.tokenizer(prompts[0] + generated_codes[0], return_tensors="pt").input_ids
        print("🧪 Full input len:", input_ids.size(1))

        # 重新计算新策略下的 log_probs（现在是整句）
        new_log_probs = self.compute_log_probs(prompts, generated_codes)

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

        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ NaN or Inf detected in PPO loss, skipping this update.")
            return

        # 反向传播 + 参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"PPO Update Done! Loss = {loss.item():.4f}")
