import torch
class TrajectoryBuffer:
    """
    Simple in-memory rollout buffer for on-policy A2C-style algorithms working
    with autoregressive language models.

    Key behaviour after FIX:
    - `state` is expected to be the FULL token sequence at the time of storage
      (prompt + generated so far).
    - `action` may be a *single token id* (scalar) OR a 1D tensor/list of the
      **newly generated tokens** for that sample. We do **not** flatten into
      scalars; the trainer will pad/stack.
    - `log_prob` must shape-match `action` (scalar ↔ scalar, sequence ↔ sequence).
    - `value` should be a scalar baseline for the *prompt state* (or the state
      at the start of generation) — whatever your trainer expects; we do not
      enforce shape beyond squeezing to 1D.
    """

    def __init__(self, auto_cpu: bool = True, debug: bool = False):
        """
        Args:
            auto_cpu: if True, incoming tensors are detached() and moved to cpu
                      when stored (saves GPU mem during rollout accumulation).
            debug:    print shape mismatches & summary info.
        """
        self.auto_cpu = auto_cpu
        self.debug = debug
        self.reset()

    # --------------------------------------------------
    def reset(self):
        """清空所有存储内容。"""
        self.states = []     # List[Tensor[T_full]]
        self.actions = []    # List[Tensor[T_gen] or scalar Tensor]
        self.rewards = []    # List[float]
        self.dones = []      # List[bool]
        self.log_probs = []  # List[Tensor[T_gen] or scalar Tensor]
        self.values = []     # List[Tensor[1] or float]

    # --------------------------------------------------
    def _to_storable_tensor(self, x):
        """Detach → cpu → long/float 按输入 dtype 保留。"""
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
            # 推断类型（若全是 int → long；否则 float）
            if all(isinstance(v, int) for v in x):
                return torch.tensor(x, dtype=torch.long)
            return torch.tensor(x, dtype=torch.float32)
        raise TypeError(f"Unsupported type for buffer storage: {type(x)}")

    # --------------------------------------------------
    def store(self, state, action, reward, done, log_prob, value):
        """
        参数:
            state (Tensor | List[int]): FULL tokenized sequence (prompt+gen).
            action (Tensor | int | List[int]): The *generated* tokens ONLY.
            reward (float): scalar reward.
            done (bool): episode terminated.
            log_prob (Tensor | float | List[float]): log-prob(s) of `action`.
            value (Tensor | float): value baseline; trainer decides how to use.

        形状约定（软约束）：
            len(state) >= len(action)   （生成不应超过总序列）
            len(action) == len(log_prob)
        """
        st = self._to_storable_tensor(state)
        ac = self._to_storable_tensor(action)
        lp = self._to_storable_tensor(log_prob)
        val = self._to_storable_tensor(value)

        # --- shape sanity ----------------------------------------------------
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

    # --------------------------------------------------
    def get_all(self):
        """
        返回 tuple (states, actions, rewards, dones, log_probs, values).
        所有字段均为 list；trainer 负责 pad & stack 到 device。
        """
        return (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.values,
        )

    # --------------------------------------------------
    def prompt_length_for(self, idx: int) -> int:
        """
        根据 state & action 长度估计 prompt token 数：
            prompt_len = len(state) - len(action)
        若 action 比 state 长（异常），返回 0。
        """
        st = self.states[idx]
        ac = self.actions[idx]
        return max(0, st.numel() - ac.numel())

    # --------------------------------------------------
    def iter_sequences(self):
        """
        Yield (state, action, prompt_len, reward, done, log_prob, value)
        方便调试与自定义 trainer。
        """
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

    # --------------------------------------------------
    def __len__(self):
        """当前缓冲中样本数。"""
        return len(self.states)

    # --------------------------------------------------
    def as_dict(self):
        """调试用：字典形式 dump。"""
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "log_probs": self.log_probs,
            "values": self.values,
        }

    # --------------------------------------------------
    def __repr__(self):
        """简洁展示 buffer 状态（含平均长度，便于排查 PPL 问题）。"""
        n = len(self)
        if n == 0:
            return "TrajectoryBuffer(size=0)"
        mean_state = sum(s.numel() for s in self.states) / n
        mean_action = sum(a.numel() for a in self.actions) / n
        return (
            f"TrajectoryBuffer(size={n}, mean_state_len={mean_state:.1f}, "
            f"mean_action_len={mean_action:.1f})"
        )
