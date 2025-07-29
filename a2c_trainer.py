import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, List, Union, Optional


class A2CTrainer:
    """
    Minimal A2C + auxiliary LM loss trainer **without wandb side-effects**.

    - 所有 wandb 相关逻辑（状态缓存、字段）已移除；只返回 metrics。
    - LM loss 仅计算 *actions* 对应 token（若提供）；否则仅 mask padding。
    - 返回的 metrics 都是 Python float，主调方可直接 wandb.log。
    """

    def __init__(
        self,
        model,
        buffer,
        reward_fn,
        device: Union[str, torch.device] = "cpu",
        gamma: float = 0.99, # (previously:pre) 0.99
        value_coef: float = 0.5, # (pre) 0.5
        entropy_coef: float = 0.01, # (pre) 0.01
        lr: float = 3e-5, # (pre) 5e-5
        pad_token_id: int = 0, # (pre)
        batch_size: int = 4,
        debug: bool = False,
        lm_weight: float = 0.7,  # (pre) 0.1
    ):
        self.model = model
        self.buffer = buffer
        self.reward_fn = reward_fn
        self.device = torch.device(device)

        # 如果外部没有明确传 pad_token_id，则尝试从模型继承
        self.pad_token_id = getattr(model, "pad_token_id", pad_token_id)

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.debug = debug
        self.lm_weight = lm_weight  # ### FIX:

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    @staticmethod
    def _to_scalar(x: Any) -> float:
        """Convert tensor / scalar-like to a safe Python float."""
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return 0.0
            x = x.detach().float().mean().item()
            if x != x:  # NaN
                return 0.0
            if x == float("inf") or x == float("-inf"):
                return 0.0
            return x
        if isinstance(x, (int, float)):
            if x != x or x == float("inf") or x == float("-inf"):
                return 0.0
            return float(x)
        return 0.0

    # ------------------------------------------------------------------
    def compute_returns(self, rewards, dones, last_value):
        """
        Standard discounted return computation (no GAE).
        rewards: list[float]
        dones:   list[bool]
        last_value: scalar or tensor
        """
        R = self._to_scalar(last_value)
        returns = []
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0
            R = float(r) + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def _prep_values(self, values_list: List[Any]) -> torch.Tensor:
        """
        Buffer values may be scalar baselines, token-level arrays, etc.
        We reduce to *final token* scalar per sequence.
        """
        vals = []
        for v in values_list:
            if isinstance(v, torch.Tensor):
                v = v.detach().float()
                if v.dim() == 0:
                    vals.append(v)
                elif v.dim() == 1:
                    vals.append(v[-1])
                else:
                    vals.append(v.view(-1)[-1])
            else:
                vals.append(torch.tensor(float(v), dtype=torch.float32))
        return torch.stack(vals).to(self.device)  # [B]

    # ------------------------------------------------------------------
    def _calc_entropy_from_logits(
        self, logits: torch.Tensor, actions: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Approximate token-level entropy using picked action log-probs.
        If shapes don't align, fall back to 0 entropy (handled in caller).
        """
        if actions is None or len(actions) == 0:
            return torch.tensor(0.0, device=logits.device)

        # pad actions
        act = pad_sequence(
            actions, batch_first=True, padding_value=self.pad_token_id
        ).to(logits.device)  # [B, Ta]

        # 如果动作序列长度 != logits 序列长度，右对齐（假设 actions 来自末尾生成段）
        if act.size(1) < logits.size(1):
            # 在左侧 pad token_id，使长度匹配
            pad_len = logits.size(1) - act.size(1)
            pad = torch.full(
                (act.size(0), pad_len), self.pad_token_id, dtype=act.dtype, device=act.device
            )
            act = torch.cat([pad, act], dim=1)
        elif act.size(1) > logits.size(1):
            act = act[:, -logits.size(1) :]  # 截尾

        # mask 指出哪些位置是真正的动作（不是 pad_token）
        act_mask = (act != self.pad_token_id).to(logits.dtype)  # [B, T]
        if act.shape[:2] != logits.shape[:2]:
            return torch.tensor(0.0, device=logits.device)

        logprobs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
        picked = logprobs.gather(-1, act.unsqueeze(-1)).squeeze(-1)  # [B, T]
        picked = picked * act_mask  # mask 掉 fake pad
        denom = act_mask.sum().clamp(min=1.0)
        ent = -(picked.sum() / denom)
        return ent

    # ------------------------------------------------------------------
    def _build_lm_labels_from_actions(
        self,
        states: torch.Tensor,
        actions: Optional[List[torch.Tensor]],
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        ### FIX: 构造 LM labels 以**只训练模型生成部分**，避免 prompt 造成虚高损失。

        假设 rollout 结构：
            states = [prompt tokens + generated tokens]
            actions = 列表（每个样本生成 token 序列）
        我们把所有标签先置为 -100（忽略），然后把 *末尾 len(actions_i)* 段置为对应 token id。
        如果没有 actions（例如纯离线训练），退回 mask padding：labels[pad]= -100。

        返回: labels shape == states shape (int64)
        """
        labels = torch.full_like(states, fill_value=-100)

        if actions is not None and len(actions) > 0:
            # pad actions 到 batch
            act = pad_sequence(
                actions, batch_first=True, padding_value=self.pad_token_id
            ).to(states.device)  # [B, Ta]

            B, T = states.shape
            Ta = act.size(1)

            # 对齐方式：将动作序列对齐到 states 的 *末尾*（假设动作是最后生成的 token）
            if Ta <= T:
                # 取 states 的最后 Ta 段；只有动作非 pad 的位置填 label
                act_mask = (act != self.pad_token_id)
                # gather states 对应末尾区间
                tgt_slice = states[:, -Ta:]
                labels[:, -Ta:] = torch.where(act_mask, tgt_slice, torch.full_like(tgt_slice, -100))
            else:
                # 动作长度超过 state（不太可能）→ 仅取末尾 T
                act = act[:, -T:]
                act_mask = (act != self.pad_token_id)
                tgt_slice = states
                labels = torch.where(act_mask, tgt_slice, torch.full_like(tgt_slice, -100))
        else:
            # 无 actions → 只忽略 padding，prompt 仍计入 LM（退化为旧逻辑）
            labels = states.clone()
            labels[attn_mask == 0] = -100

        return labels

    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, float]:
        """
        One A2C update over *all* transitions currently in buffer.
        Returns python-float metrics dict (safe for wandb in caller).
        """
        if len(self.buffer) == 0:
            # 空 buffer：返回 0 metrics，主调方依旧可 log 全字段
            return {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "lm_loss": 0.0,
                "adv_mean": 0.0,
                "return_mean": 0.0,
                "log_prob_mean": 0.0,
            }

        # --------------------------------------------------
        # 1. unpack buffer
        #    约定：states=list[tensor[int]], actions=list[tensor[int]] ...
        # --------------------------------------------------
        states, actions, rewards, dones, log_probs, values = self.buffer.get_all()

        # pad to batch
        states = pad_sequence(
            states, batch_first=True, padding_value=self.pad_token_id
        ).to(self.device)  # [B, T]

        # 截断 block_size（防止越界位置嵌入）
        block_size = getattr(self.model.model.config, "block_size", states.size(1))
        if states.size(1) > block_size:
            states = states[:, :block_size]

        # attention mask (real tokens)
        attn_mask = (states != self.pad_token_id).long()

        # rewards / dones
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # rollout log_probs（token级？序列级？）→ reduce 序列级
        if isinstance(log_probs, torch.Tensor):
            lp = log_probs.to(self.device)
            if lp.dim() == 1:
                seq_log_probs = lp  # already [B]
            else:
                seq_log_probs = lp.sum(dim=-1)  # [B]
        else:
            lp = pad_sequence(log_probs, batch_first=True, padding_value=0.0).to(self.device)
            seq_log_probs = lp.sum(dim=-1)  # [B]

        # value baseline
        values_t = self._prep_values(values)  # [B]

        # --------------------------------------------------
        # 2. bootstrap last value（无梯度；取 batch 最后一条）
        # --------------------------------------------------
        with torch.no_grad():
            out_boot = self.model(states[-1:], attention_mask=attn_mask[-1:])
            last_value = out_boot["value"].float().mean()  # scalar-ish
        last_value_scalar = last_value.item()

        # --------------------------------------------------
        # 3. 构造 LM labels：只训练 actions 段（降低 PPL）
        # --------------------------------------------------
        labels = self._build_lm_labels_from_actions(states, actions, attn_mask)

        # forward 计算 logits / lm_loss
        model_outputs = self.model(
            states, attention_mask=attn_mask, labels=labels
        )
        logits = model_outputs["logits"]
        lm_loss = model_outputs.get("lm_loss") or model_outputs.get("loss")
        if lm_loss is None:
            # 极端 fallback：自己算
            lm_loss = self.model._compute_lm_loss(logits, labels)

        # --------------------------------------------------
        # 4. returns & advantages
        # --------------------------------------------------
        returns = self.compute_returns(rewards, dones, last_value_scalar).to(self.device)  # [B]
        advantages = returns - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --------------------------------------------------
        # 5. policy/value/entropy/total loss
        # --------------------------------------------------
        policy_loss = -(seq_log_probs * advantages.detach()).mean()
        value_loss = self.criterion(values_t, returns)

        if actions is not None and len(actions) > 0:
            entropy = self._calc_entropy_from_logits(logits, actions)
        else:
            entropy = -seq_log_probs.mean()  # fallback

        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
            + self.lm_weight * lm_loss
        )

        # --------------------------------------------------
        # 6. backward & step
        # --------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # --------------------------------------------------
        # 7. 清空 buffer
        # --------------------------------------------------
        self.buffer.reset()

        # --------------------------------------------------
        # 8. metrics (python floats)
        # --------------------------------------------------
        metrics = {
            "total_loss": self._to_scalar(loss),
            "policy_loss": self._to_scalar(policy_loss),
            "value_loss": self._to_scalar(value_loss),
            "entropy": self._to_scalar(entropy),
            "lm_loss": self._to_scalar(lm_loss),
            "adv_mean": self._to_scalar(advantages.mean()),
            "return_mean": self._to_scalar(returns.mean()),
            "log_prob_mean": self._to_scalar(seq_log_probs.mean()),
        }

        if self.debug:
            non_pad = attn_mask.sum().item()
            total_tok = attn_mask.numel()
            eff = non_pad / max(total_tok, 1)
            # 在 actions 模式下，真实用于 LM 的 token 数：
            lm_tok = (labels != -100).sum().item()
            print(
                f"[DEBUG A2CTrainer] eff_tokens={eff:.2%} | lm_tokens={lm_tok} | "
                f"loss={metrics['total_loss']:.4f} | lm={metrics['lm_loss']:.4f}"
            )

        return metrics
