import torch
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, model, optimizer, buffer, clip_epsilon=0.2):
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.clip_epsilon = clip_epsilon
        self.tokenizer = model.tokenizer  # 使用 model 自带的 tokenizer

    def ppo_loss(self, old_log_probs, new_log_probs, rewards):
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
        return loss

    def compute_log_probs(self, inputs, actions): # 适配2维
        # Tokenize inputs and actions
        input_encodings = self.tokenizer(list(inputs), return_tensors="pt", padding=True, truncation=True)
        action_encodings = self.tokenizer(list(actions), return_tensors="pt", padding=True, truncation=True)

        input_ids = input_encodings.input_ids  # [batch, prompt_len]
        attention_mask = input_encodings.attention_mask
        action_ids = action_encodings.input_ids  # [batch, action_len]

        # model.forward returns logits with shape [batch, vocab_size]
        logits = self.model(input_ids, attention_mask=attention_mask)
        if isinstance(logits, tuple):  # if (logits, loss)
            logits = logits[0]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [batch, vocab_size]

        # 只对最后一个 token 计算 log_prob（因为 logits 是 2维的）
        last_token_ids = action_ids[:, -1]  # [batch]
        log_prob = log_probs.gather(1, last_token_ids.unsqueeze(-1)).squeeze(-1)  # [batch]

        return log_prob  # 每个样本一个log_prob

    def update(self, buffer, batch_size=None):
        actual_batch_size = len(buffer) if batch_size is None else min(batch_size, len(buffer))

        if actual_batch_size == 0:
            print("⚠️ PPO skipped: not enough data in buffer.")
            return

        prompts, generated_codes, rewards, old_log_probs = buffer.sample(actual_batch_size)

        # 计算 PPO loss 并优化
        new_log_probs = self.compute_log_probs(prompts, generated_codes)
        loss = self.ppo_loss(old_log_probs, new_log_probs, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"✅ PPO Update Done! Loss = {loss.item():.4f}")

