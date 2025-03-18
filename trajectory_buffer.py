import torch
class TrajectoryBuffer:
# Initialization
    def __int__(self, max_size = 100):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.max_size = max_size

# Storage Tracks: When the buffer exceeds max_size, the oldest data is deleted to ensure that the buffer size is fixed.
    def store(self, state, action, reward, next_state):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)

# Randomly sampled trajectory
    def sample(self, batch_size = 32):
        indices = torch.randint(0, len(self.states), (batch_size,))
        return (
            torch.stack([self.states[i] for i in indices]),
            torch.stack([self.actions[i] for i in indices]),
            torch.tensor([self.rewards[i] for i in indices]),
            torch.stack([self.next_states[i] for i in indices]),
        )

