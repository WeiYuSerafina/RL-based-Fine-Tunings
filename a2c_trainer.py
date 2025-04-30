import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

class A2CTrainer:
    def __init__(self, model, buffer, reward_fn, device='cpu',
                 gamma=0.99, value_coef=0.5, entropy_coef=0.01, lr=1e-4, pad_token_id=0):
        self.model = model
        self.buffer = buffer
        self.reward_fn = reward_fn
        self.device = device
        self.pad_token_id = pad_token_id

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def train_step(self):
        states, actions, rewards, dones, log_probs, values = self.buffer.get_all()
        states = pad_sequence(states, batch_first=True, padding_value=self.pad_token_id).to(self.device)
        actions = torch.tensor([a.item() for a in actions], dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)

        # 只这里保留 no_grad（因为 last_value 不参与梯度计算）
        with torch.no_grad():
            _, last_value = self.model(states[-1].unsqueeze(0))
        last_value = last_value.squeeze()

        # compute returns
        returns = self.compute_returns(rewards, dones, last_value)
        returns = torch.tensor(returns).to(self.device)

        # compute advantages
        advantages = returns - values

        # 正确保留计算图
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = self.criterion(values, returns)
        entropy = -(log_probs * log_probs.exp()).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()  # 这里不会再报错
        self.optimizer.step()

        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }
