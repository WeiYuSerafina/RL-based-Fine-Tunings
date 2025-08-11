import os
import json
import time
import random
import argparse

import torch
import numpy as np
import wandb

from nano_gpt_ppo_policy import NanoGPTPolicy
from ppo_trainer import PPOTrainer
from trajectory_buffer import TrajectoryBuffer
from reward_function import reward_function
from dataset_loader import MBPPDataset
from ppo_trainer import PPOTrainer

from evaluate_ppo_a2c_perplexity import evaluate_perplexity,load_prompt_completion_pairs
import csv, os

def log_ppl_to_csv(step: int, ppl: float, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "ppl"])
        w.writerow([step, ppl])

# Sweep config
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

# Main training loop
from types import SimpleNamespace

def train_loop(cfg: SimpleNamespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model & tokenizer
    model_name = "./out/mbpp_baseline_v3"
    tokenizer_path = "./data/mbpp_new"

    model = NanoGPTPolicy(model_name, tokenizer_path=tokenizer_path)
    tokenizer = model.tokenizer

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                  total_iters=cfg.total_steps)

    # helpers
    buffer = TrajectoryBuffer()
    ppo = PPOTrainer(model, tokenizer, optimizer, buffer, clip_epsilon=0.2, config=cfg)

    # load dataset
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

        # PPO update
        if len(buffer) >= cfg.batch_size:
            loss = ppo.update(buffer)
            buffer.clear()
            scheduler.step()

        # moving average reward
        recent_rewards.append(reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_recent_reward = float(np.mean(recent_rewards))

        # logging
        if step % cfg.log_interval == 0:
            wandb.log(
                {
                "steo": step,
                "reward": reward,
                "moving_avg_reward": avg_recent_reward,
                "lr": optimizer.param_groups[0]['lr'],
                **getattr(ppo, "last_stats", {}),
                },
                step=step,
            )

        # evaluation print
        if step % cfg.eval_interval == 0:
            print(f"[Step {step}] AvgReward={avg_recent_reward:.4f} | Reward={reward:.4f}")

        # evaluate PPL + write to CSV
        try:
            eval_pairs = load_prompt_completion_pairs(
                path='google-research/mbpp/sanitized-mbpp.json',
                max_samples=50
            )

            ppl_val = evaluate_perplexity(
                model=model,
                tokenizer=tokenizer,
                prompt_full_pairs=eval_pairs,
                batch_size=8,
                max_length=256,
            )

            log_ppl_to_csv(step, ppl_val, "./logs/ppl_ppo.csv")
            wandb.log({"ppl": ppl_val}, step=step)
            print(f"[Step {step}] PPL(PPO) = {ppl_val:.2f}")

        except Exception as e:
            print(f"[Warning] Failed to evaluate PPL at step {step}: {e}")

        # early stopping
        if avg_recent_reward > best_avg_reward:
            best_avg_reward, best_step = avg_recent_reward, step
            best_state = model.model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= cfg.early_stop_patience:
            print(f"Early stopped at step {step}, moving_avg_reward no longer improves.")
            break

    # save best
    out_dir = f"./saved_nanoGPT_finetuned/PPO_best_step_{best_step}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "pytorch_model.bin"))

    # write the underlying GPT config as JSON
    gpt_cfg_dict = model.model.config.__dict__  # GPTConfig 转字典
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(gpt_cfg_dict, f, indent=2)

    # save tokenizer
    tokenizer.save_pretrained(out_dir)
    print(f"Best model saved to {out_dir} (step={best_step}, avg_reward={best_avg_reward:.4f})")

# CLI
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
            "total_steps": 2000,
            "log_interval": 10,
        }
    )
    cfg = SimpleNamespace(**wandb.config)
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
