import torch
import os
import json
import wandb
import argparse
from nano_gpt_policy import NanoGPTPolicy
from ppo_trainer import PPOTrainer
from trajectory_buffer import TrajectoryBuffer
from reward_function import reward_function
from dataset_loader import ArcadeDataset
from train_ppo import train_ppo

# === Sweep 参数设置 ===
sweep_config = {
    "method": "random",
    "metric": {"name": "moving_avg_reward", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 1e-6, "max": 1e-4},
        "batch_size": {"values": [8, 16, 32]},
        "max_new_tokens": {"values": [64, 100, 128]},
        "early_stop_patience": {"values": [200, 300, 400]},
        "ppo_epochs": {"values": [2, 4, 8]}
    }
}

# === 主训练函数 ===
from types import SimpleNamespace

def main():
    wandb.init(
        project="nanoGPT-RL-PPO",
        config={
            "lr": 2e-5,
            "batch_size": 8,
            "max_new_tokens": 100,
            "early_stop_patience": 100,
            "eval_interval": 100,
            "ppo_epochs": 4
        }
    )
    config = SimpleNamespace(**dict(wandb.config))

    train_ppo(config)

    model_name = "./saved_nanoGPT"
    model = NanoGPTPolicy(model_name)
    tokenizer = model.tokenizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    buffer = TrajectoryBuffer()
    ppo = PPOTrainer(model, tokenizer, optimizer, buffer)

    dataset_path = 'arcade-nl2code/arcade_nl2code/annotated_dataset/merged_dataset_new_tasks_cleaned_v2.jsonl'
    dataset = ArcadeDataset(dataset_path)

    best_model_state = None
    best_step = -1
    best_avg_reward = -float("inf")
    early_stop_counter = 0
    recent_rewards = []

    for step in range(1000):
        prompt, ground_truth = dataset.sample()
        input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids

        output_ids, all_log_probs = model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens
        )

        if torch.isnan(all_log_probs).any():
            print("Detected NaN in log_probs, skipping this step.")
            continue

        new_tokens = output_ids[:, input_ids.shape[1]:]
        generated_code = model.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        if "ructionruction" in generated_code:
            reward = 0.0
        else:
            reward = reward_function(generated_code, ground_truth, prompt=prompt)
            if isinstance(reward, float) and (reward != reward or reward == float("inf") or reward == float("-inf")):
                print("\u274c Invalid reward (NaN or Inf), skipping this step.")
                continue

        avg_log_prob = all_log_probs.mean().item()
        buffer.add(prompt, generated_code, reward, avg_log_prob)

        if step > 0 and len(buffer) > 0:
            loss = ppo.update(buffer)
            if loss is None:
                print("Skipping PPO update due to NaN or invalid data.")
            buffer.clear()

        recent_rewards.append(reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_recent_reward = sum(recent_rewards) / len(recent_rewards)

        wandb.log({
            "step": step,
            "reward": reward,
            "avg_log_prob": avg_log_prob,
            "moving_avg_reward": avg_recent_reward,
            "generated_code": generated_code
        })

        if avg_recent_reward > best_avg_reward:
            best_avg_reward = avg_recent_reward
            best_model_state = model.model.state_dict()
            best_step = step
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if step % 100 == 0:
            print(f"Step {step}")
            print("Prompt:\n", prompt)
            print(f"Generated:\n{generated_code}")
            print(f"Reward = {reward:.4f}, Avg Log Prob = {avg_log_prob:.4f}")
            print(f"Moving Avg Reward = {avg_recent_reward:.4f}")

        if early_stop_counter >= config.early_stop_patience:
            print(f"Early stopped at step {step}, no improvement.")
            break

    if best_model_state is not None:
        best_path = f"./saved_nanoGPT_finetuned/PPO_best_step_{best_step}"
        os.makedirs(best_path, exist_ok=True)
        torch.save(best_model_state, f"{best_path}/pytorch_model.bin")
        torch.save(model, f"{best_path}/ppo_best_model.pt")
        with open(f"{best_path}/config.json", "w") as f:
            json.dump(model.model.config.__dict__, f, indent=4)
        if model.tokenizer:
            model.tokenizer.save_pretrained(best_path)
        print(f"Best model saved to: {best_path}, from step {best_step}, Avg Reward = {best_avg_reward:.4f}")

# === 启动方式支持 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run sweep agent")
    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="nanoGPT-RL-PPO")
        wandb.agent(sweep_id, function=main)
    else:
        main()
