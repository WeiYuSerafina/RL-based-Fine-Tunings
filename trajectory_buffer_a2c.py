import torch
class TrajectoryBuffer:
    """
    Simple in-memory rollout buffer for on-policy A2C-style algorithms working
    with autoregressive language models.
    """

    def __init__(self, auto_cpu: bool = True, debug: bool = False):
        self.auto_cpu = auto_cpu
        self.debug = debug
        self.reset()

    def reset(self):
        self.states = []     # List[Tensor[T_full]]
        self.actions = []    # List[Tensor[T_gen] or scalar Tensor]
        self.rewards = []    # List[float]
        self.dones = []      # List[bool]
        self.log_probs = []  # List[Tensor[T_gen] or scalar Tensor]
        self.values = []     # List[Tensor[1] or float]

    def _to_storable_tensor(self, x):
        if isinstance(x, torch.Tensor):
            t = x.detach()
            if self.auto_cpu:
                t = t.cpu()
            return t
        # ints / floats → tensor
        if isinstance(x, int):
            return torch.tensor([x], dtype=torch.long)
        if isinstance(x, float):
            return torch.tensor([x], dtype=torch.float32)
        # list/tuple
        if isinstance(x, (list, tuple)):
            if all(isinstance(v, int) for v in x):
                return torch.tensor(x, dtype=torch.long)
            return torch.tensor(x, dtype=torch.float32)
        raise TypeError(f"Unsupported type for buffer storage: {type(x)}")

    def store(self, state, action, reward, done, log_prob, value):
        st = self._to_storable_tensor(state)
        ac = self._to_storable_tensor(action)
        lp = self._to_storable_tensor(log_prob)
        val = self._to_storable_tensor(value)

        # shape sanity
        if ac.dim() == 2 and ac.size(0) == 1:
            ac = ac.squeeze(0)
        if lp.dim() == 2 and lp.size(0) == 1:
            lp = lp.squeeze(0)

        # logprob shape match
        if ac.numel() != lp.numel() and self.debug:
            print(
                f"[TrajectoryBuffer][WARN] action len {ac.numel()} != log_prob len {lp.numel()}."
                " Will broadcast min length on trainer side."
            )

        # action length sanity
        if ac.numel() > st.numel() and self.debug:
            print(
                f"[TrajectoryBuffer][WARN] action longer than state "
                f"({ac.numel()} > {st.numel()}) — check rollout logic."
            )

        self.states.append(st)
        self.actions.append(ac)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(lp)
        self.values.append(val)

    def get_all(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.values,
        )

    def prompt_length_for(self, idx: int) -> int:
        st = self.states[idx]
        ac = self.actions[idx]
        return max(0, st.numel() - ac.numel())

    def iter_sequences(self):
        for i in range(len(self.states)):
            yield (
                self.states[i],
                self.actions[i],
                self.prompt_length_for(i),
                self.rewards[i],
                self.dones[i],
                self.log_probs[i],
                self.values[i],
            )

    def __len__(self):
        return len(self.states)

    def as_dict(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "log_probs": self.log_probs,
            "values": self.values,
        }

    def __repr__(self):
        n = len(self)
        if n == 0:
            return "TrajectoryBuffer(size=0)"
        mean_state = sum(s.numel() for s in self.states) / n
        mean_action = sum(a.numel() for a in self.actions) / n
        return (
            f"TrajectoryBuffer(size={n}, mean_state_len={mean_state:.1f}, "
            f"mean_action_len={mean_action:.1f})"
        )
