import os
import json
import time
import random
import argparse
from types import SimpleNamespace

import torch
import numpy as np
import wandb

from nano_gpt_ppo_policy import NanoGPTPolicy
from ppo_trainer import PPOTrainer
from trajectory_buffer import TrajectoryBuffer
from reward_function import reward_function
from dataset_loader import MBPPDataset


# === Sweep config ===
sweep_config = {
    "method": "random",
    "metric": {"name": "moving_avg_reward", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 1e-6, "max": 1e-4},
        "batch_size": {"values": [8, 16, 32]},
        "max_new_tokens": {"values": [64, 100, 128]},
        "early_stop_patience": {"values": [200, 300, 400]},
        "ppo_epochs": {"values": [2, 4, 8]},
        "eval_interval": {"values": [50, 100]},
        "total_steps": {"value": 5000},
        "log_interval": {"value": 10}
    }
}

# === utils ===
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === main training loop ===
from types import SimpleNamespace

def train_loop(cfg: SimpleNamespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model & tokenizer ---
    model_name = "./out/mbpp_baseline_v2"
    tokenizer_path = "./data/mbpp_new"

    model = NanoGPTPolicy(model_name, tokenizer_path=tokenizer_path)
    tokenizer = model.tokenizer

    # --- optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                  total_iters=cfg.total_steps)

    # --- helpers ---
    buffer = TrajectoryBuffer()
    ppo = PPOTrainer(model, tokenizer, optimizer, buffer, clip_epsilon=0.2, config=cfg)

    # --- load dataset ---
    dataset_path = 'google-research/mbpp/mbpp_train.jsonl'
    dataset = MBPPDataset(dataset_path)

    best_avg_reward = -float("inf")
    best_step = -1
    early_stop_counter = 0
    recent_rewards = []

    for step in range(cfg.total_steps):
        prompt, ground_truth = dataset.sample()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output_ids, log_probs = model.generate(input_ids, max_new_tokens=cfg.max_new_tokens)

        new_tokens = output_ids[:, input_ids.shape[1]:]
        gen_code = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        reward = reward_function(gen_code, ground_truth, prompt=prompt)
        if not np.isfinite(reward):
            continue  # skip invalid reward
        reward = float(np.clip(reward, -10.0, 10.0))

        buffer.add(prompt, gen_code, reward, log_probs.mean().item())

        # --- PPO update ---
        if len(buffer) >= cfg.batch_size:
            loss = ppo.update(buffer)
            buffer.clear()
            scheduler.step()

        # --- moving average reward ---
        recent_rewards.append(reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_recent_reward = float(np.mean(recent_rewards))

        # --- logging ---
        if step % cfg.log_interval == 0:
            wandb.log({
                "step": step,
                "reward": reward,
                "moving_avg_reward": avg_recent_reward,
                "lr": optimizer.param_groups[0]['lr']
            })

        # --- evaluation print ---
        if step % cfg.eval_interval == 0:
            print(f"[Step {step}] AvgReward={avg_recent_reward:.4f} | Reward={reward:.4f}")

        # --- early stopping ---
        if avg_recent_reward > best_avg_reward:
            best_avg_reward, best_step = avg_recent_reward, step
            best_state = model.model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= cfg.early_stop_patience:
            print(f"Early stopped at step {step}, moving_avg_reward no longer improves.")
            break

    # --- save best ---
    out_dir = f"./saved_nanoGPT_finetuned/PPO_best_step_{best_step}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "pytorch_model.bin"))

    # === 把底层 GPT 的 config 写成 JSON ===
    gpt_cfg_dict = model.model.config.__dict__  # GPTConfig 转字典
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(gpt_cfg_dict, f, indent=2)

    # save tokenizer
    tokenizer.save_pretrained(out_dir)
    print(f"Best model saved to {out_dir} (step={best_step}, avg_reward={best_avg_reward:.4f})")

# === Command-Line Interface (CLI) ===
def main():
    wandb.init(
        project="nanoGPT-RL-PPO",
        config={
            "lr": 1e-5,
            "batch_size": 8,
            "max_new_tokens": 100,
            "early_stop_patience": 500,
            "eval_interval": 100,
            "ppo_epochs": 4,
            "total_steps": 3000,
            "log_interval": 10,
            "seed": 42
        }
    )
    cfg = SimpleNamespace(**wandb.config)
    set_seed(cfg.seed)
    train_loop(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run wandb sweep agent")
    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="nanoGPT-RL-PPO")
        wandb.agent(sweep_id, function=main)
    else:
        main()
