import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trajectory_buffer_a2c import TrajectoryBuffer
from reward_function import reward_function

def train(model, dataset, trainer, tokenizer, device, debug=True):
    buffer = trainer.buffer
    model.eval()
    step = 0
    max_new_tokens = 10  # ✅ 每次生成 10 个 token

    for epoch in range(5):
        for sample in dataset:
            step += 1
            prompt = sample["prompt"]

            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]  # shape: [1, T]

            # 初始状态输入，获取初始 value
            _, value = model(**inputs)

            # 初始化生成 token 序列
            generated_ids = input_ids[0].clone()
            all_log_probs = []

            for _ in range(max_new_tokens):
                logits, _ = model(generated_ids.unsqueeze(0))  # shape: [1, T, V]
                last_logits = logits[:, -1, :]  # shape: [1, V]
                probs = torch.softmax(last_logits, dim=-1).squeeze(0)  # shape: [V]

                dist = torch.distributions.Categorical(probs)
                action = dist.sample()               # shape: scalar
                log_prob = dist.log_prob(action)     # shape: scalar

                generated_ids = torch.cat([generated_ids, action.unsqueeze(0)], dim=0)
                all_log_probs.append(log_prob)

            # 解码输出文本
            decoded_output = tokenizer.decode(generated_ids.tolist())

            # 奖励评估
            reward = reward_function(prompt, decoded_output)
            done = True

            # 平均 log_prob（作为 policy loss 输入）
            avg_log_prob = torch.stack(all_log_probs).mean()

            # 存入 buffer：将最后 action 存入即可
            buffer.store(input_ids.squeeze(0), action, reward, done, avg_log_prob, value)

            # Debug 输出
            if debug and step % 50 == 0:
                print(f"\n[Step {step}]")
                print("Prompt:\n", prompt)
                print("Generated:\n", decoded_output)
                print(f"Reward: {reward:.4f}, Avg Log Prob: {avg_log_prob.item():.4f}")

        # 策略更新
        logs = trainer.train_step()
        print(f"Epoch {epoch}: {logs}")
        buffer.reset()
