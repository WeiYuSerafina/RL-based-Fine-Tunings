import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class PPOTrainer:
    # Initialize
    def __init__(self, model, tokenizer, buffer, lr = 1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    # loss function
    def ppo_loss(self, old_probs, new_probs, rewards, epsilon = 0.2):
        ratio = torch.exp(new_probs - old_probs)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
        return loss

    # train step
    def train_step(self, batch_size = 32):
        states, actions, rewards, next_states = self.buffer.sample(batch_size)
        old_probs = self.model(states).log_prob(actions)

        self.optimizer.zero.grad()
        new_probs = self.model(next_states).log_prob(actions)
        loss = self.ppo_loss(old_probs, new_probs, rewards)
        loss.backward()
        self.optimizer.step()

        return loss.item()


