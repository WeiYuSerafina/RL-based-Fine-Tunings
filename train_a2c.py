# a2c_rollout.py  (示例文件名；与 run_a2c.py 保持 import 一致)
import torch
from typing import List, Optional
from trajectory_buffer_a2c import TrajectoryBuffer
from reward_function import reward_function


def train(
    model,
    dataset,
    trainer,
    tokenizer,
    device: torch.device,
    *,
    num_epochs: int = 4,
    max_new_tokens: int = 16,
    debug: bool = True,
):
    """
    Minimal rollout loop for on-policy A2C fine-tuning.

    After FIX:
      • Collect full generated token seq (actions) per sample.
      • Slice out *generated-only* text for reward_function().
      • Call reward_function with correct arg names/order.
      • No wandb here (centralized in run_a2c.py).
    """
    buffer: TrajectoryBuffer = trainer.buffer
    model.eval()  # rollout w/o grad
    step = 0

    for epoch in range(num_epochs):
        for sample in dataset:
            step += 1
            prompt: str = sample["prompt"]

            # --------------------------------------------------
            # 1) Encode prompt
            # --------------------------------------------------
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,  # ### FIX: 保持与训练一致，避免多余 BOS/EOS
            )
            inputs = {k: v.to(device) for k, v in enc.items()}
            input_ids = inputs["input_ids"]               # [1, T_prompt]
            attention_mask = inputs.get("attention_mask") # [1, T_prompt] or None
            prompt_len = input_ids.size(1)                # ### FIX: 记录 prompt token 长度

            # --------------------------------------------------
            # 2) Baseline value for prompt state
            # --------------------------------------------------
            with torch.no_grad():
                out0 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                )
                value0 = out0["value"]  # [1, 1]

            # --------------------------------------------------
            # 3) Autoregressive sampling
            # --------------------------------------------------
            generated_ids = input_ids[0].clone().to(device)  # flatten → [T]
            gen_actions: List[torch.Tensor] = []             # ### FIX: 收集所有新 token
            gen_log_probs: List[torch.Tensor] = []           # ### FIX: 同步 logprob

            for _ in range(max_new_tokens):
                out = model(
                    input_ids=generated_ids.unsqueeze(0),
                    attention_mask=None,  # 自行推导；序列随生成增长
                    labels=None,
                )
                logits = out["logits"]                      # [1, cur_T, V]
                probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze(0)  # [V]

                dist = torch.distributions.Categorical(probs)
                action = dist.sample()                      # scalar token id
                logp = dist.log_prob(action)                # scalar logprob

                gen_actions.append(action)                  # ### FIX: push
                gen_log_probs.append(logp)                  # ### FIX: push

                generated_ids = torch.cat([generated_ids, action.unsqueeze(0)], dim=0)

                # early stop on EOS
                if action.item() == tokenizer.eos_token_id:
                    break

            # --------------------------------------------------
            # 4) Decode
            # --------------------------------------------------
            decoded_full = tokenizer.decode(generated_ids.tolist())
            # ### FIX: 只取生成段用于 reward
            generated_code_only = tokenizer.decode(
                generated_ids[prompt_len:].tolist()
            )

            # --------------------------------------------------
            # 5) Reference code（来自数据集 sample）
            # --------------------------------------------------
            # ### FIX: 兼容 MBPP 字段名
            reference_code = (
                sample.get("code")
                or sample.get("completion")
                or ""
            )

            # --------------------------------------------------
            # 6) Reward (正确参数顺序/命名)
            # --------------------------------------------------
            reward = reward_function(
                generated_code=generated_code_only,  # ### FIX
                reference_code=reference_code,       # ### FIX
                prompt=prompt,                       # ### FIX
            )
            done = True

            # 平均 logprob（仅日志用；策略梯度用 token 列表）
            if gen_log_probs:
                avg_log_prob = torch.stack(gen_log_probs).mean()
            else:
                avg_log_prob = torch.tensor(0.0, device=device)

            # --------------------------------------------------
            # 7) Push to buffer
            # --------------------------------------------------
            full_state_cpu = generated_ids.cpu()            # prompt+gen
            if gen_actions:
                actions_cpu = torch.stack(gen_actions).cpu()   # [T_gen]
                logps_cpu   = torch.stack(gen_log_probs).cpu() # [T_gen]
            else:
                # 空生成（很少发生）
                actions_cpu = torch.tensor([tokenizer.eos_token_id])
                logps_cpu   = torch.tensor([0.0])

            buffer.store(
                full_state_cpu,
                actions_cpu,
                reward,
                done,
                logps_cpu,
                value0.squeeze(0).cpu(),   # baseline from prompt
            )

            # --------------------------------------------------
            # 8) Debug print
            # --------------------------------------------------
            if debug and step % 50 == 0:
                print(
                    f"\n[Rollout Step {step}] "
                    f"Reward={reward:.4f}, GenLen={len(gen_actions)}, AvgLogP={avg_log_prob.item():.4f}"
                )
                if debug and step % 200 == 0:
                    print("Prompt:\n", prompt)
                    print("Generated (full):\n", decoded_full)
                    print("Generated (gen-only):\n", generated_code_only)
                    print("Reference:\n", reference_code)

        # --------------------------------------------------
        # 9) policy/value update after epoch
        # --------------------------------------------------
        logs = trainer.train_step()  # trainer 内部 reset buffer

        # ### FIX: 删掉重复 buffer.reset() —— 已在 trainer.train_step() 中完成
        # buffer.reset()

        # 简单 stdout 日志（wandb 在 run_a2c.py）
        if logs:
            line = " | ".join(
                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in logs.items()
            )
            print(f"[Epoch {epoch}] {line}")
        else:
            print(f"[Epoch {epoch}] trainer returned no logs")
