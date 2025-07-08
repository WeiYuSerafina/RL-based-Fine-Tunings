import torch

class TrajectoryBuffer:
    def __init__(self, max_size=500):
        self.prompts = []
        self.generated_codes = []
        self.rewards = []
        self.log_probs = []
        self.max_size = max_size

    # add：save prompt, generated_code, reward, log_prob
    def add(self, prompt, generated_code, reward, log_prob):
        if len(self.prompts) >= self.max_size:
            self.prompts.pop(0)
            self.generated_codes.pop(0)
            self.rewards.pop(0)
            self.log_probs.pop(0)
        self.prompts.append(prompt)
        self.generated_codes.append(generated_code)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    # sample data (Not used yet, reserved)
    def sample(self, batch_size=8):
        actual_batch_size = min(batch_size, len(self.prompts))
        indices = torch.randint(0, len(self.prompts), (actual_batch_size,))

        # 将张量索引 i.item() 转成 Python int
        sampled_prompts = [self.prompts[i.item()] for i in indices]
        sampled_codes = [self.generated_codes[i.item()] for i in indices]

        # 显式 dtype=torch.float32，避免与 new_log_probs 精度不一致
        sampled_rewards = torch.tensor(
            [self.rewards[i.item()] for i in indices], dtype=torch.float32
        )
        sampled_log_probs = torch.tensor(
            [self.log_probs[i.item()] for i in indices], dtype=torch.float32
        )

        return sampled_prompts, sampled_codes, sampled_rewards, sampled_log_probs

    # clear buffer
    def clear(self):
        self.prompts.clear()
        self.generated_codes.clear()
        self.rewards.clear()
        self.log_probs.clear()

    # make `TrajectoryBuffer` compatible with the built-in `len()` function for easy size checking
    def __len__(self):
        return len(self.prompts)

