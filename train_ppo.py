import torch
import torch.optim as optim
from transformers import AutoTokenizer
from trajectory_buffer import TrajectoryBuffer
from ppo_trainer import PPOTrainer
from reward_function import reward_function
from dataset_loader import ArcadeDataset
from nano_gpt_policy import NanoGPTPolicy

# === 1. 模型与组件初始化 ===
tokenizer = AutoTokenizer.from_pretrained("data/arcade_new")
model = NanoGPTPolicy("saved_nanoGPT")
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
buffer = TrajectoryBuffer(max_size=500)
trainer = PPOTrainer(model, tokenizer, optimizer, buffer)
dataset = ArcadeDataset("arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v2.jsonl")

# === 2. 用当前策略填充 buffer（采样 + 计算 reward + 存入）===
batch_size = 8
num_rollouts = 12

for _ in range(num_rollouts):
    samples = [dataset.sample() for _ in range(batch_size)]
    prompts = [s[0] for s in samples]
    ground_truths = [s[1] for s in samples]

    # 删除 Top-K, Top-P, do_sample 的参数传递
    generated_texts, prompts, generated_ids = trainer.rollout(prompts)

    # 使用当前策略重新计算 log_probs（整句级别）
    log_probs = trainer.compute_log_probs(prompts, generated_texts)
    if torch.isnan(log_probs).any():
        print("⚠️ Detected NaN in log_probs, skipping this rollout.")
        continue

    rewards = [reward_function(gen, gt) for gen, gt in zip(generated_texts, ground_truths)]

    if any((not isinstance(r, float)) or torch.isnan(torch.tensor(r)) or torch.isinf(torch.tensor(r)) for r in rewards):
        print("⚠️ Invalid reward detected, skipping this rollout.")
        continue

    for prompt, gen, reward, log_prob in zip(prompts, generated_texts, rewards, log_probs.tolist()):
        buffer.add(prompt, gen, reward, log_prob)

print(f"✅ Buffer 填充完成，当前存储数量：{len(buffer)}")

# === 3. PPO 训练主循环（更新策略、评估、Early Stop）===
def train_ppo(config):
    # 使用 getattr 获取字段，带默认值（更安全）
    patience = getattr(config, "early_stop_patience", 10)
    eval_interval = getattr(config, "eval_interval", 100)
    best_avg_reward = float("-inf")
    no_improve_count = 0

    for epoch in range(1000):
        print(f"🔁 Epoch {epoch + 1}")
        loss = trainer.update(buffer, batch_size=8)
        if loss is None or torch.isnan(torch.tensor(loss)):
            print(f"⚠️ Epoch {epoch + 1} skipped due to NaN loss.")
            continue

        # === 4. 定期评估 PPO 策略 ===
        if (epoch + 1) % eval_interval == 0:
            eval_prompts = [dataset.sample()[0] for _ in range(16)]
            generated_texts, _ = trainer.rollout(eval_prompts)  # ✅ 已移除 Top-* 策略
            ground_truths = [dataset.lookup_ground_truth(p) for p in eval_prompts]
            rewards = [reward_function(gen, gt) for gen, gt in zip(generated_texts, ground_truths)]
            avg_reward = sum(rewards) / len(rewards)

            print(f"📊 Evaluation @ step {epoch + 1}: avg_reward = {avg_reward:.4f}")

            # === 5. 判断是否保存最优模型 ===
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                no_improve_count = 0
                print("🔼 Reward improved, saving model...")

                save_path = f"saved_nanoGPT_finetuned/PPO_best_step_{epoch + 1}"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"💾 Model checkpoint saved to: {save_path}")
            else:
                no_improve_count += 1
                print(f"⚠️ No improvement for {no_improve_count} evals.")

            # === 6. Early Stopping 判断 ===
            if no_improve_count >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} steps.")
                break
