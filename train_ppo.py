import torch, random, numpy as np
import torch.optim as optim
import wandb
from transformers import AutoTokenizer
from trajectory_buffer import TrajectoryBuffer
from ppo_trainer      import PPOTrainer
from reward_function  import reward_function
from dataset_loader   import MBPPDataset
from nano_gpt_ppo_policy import NanoGPTPolicy

# 1. Initialization
tokenizer = AutoTokenizer.from_pretrained("nanoGPT-RL/data/mbpp_new")
model     = NanoGPTPolicy("nanoGPT-RL/out/mbpp_baseline_v2")

optimizer = optim.AdamW(model.parameters(), lr=1e-5)
buffer    = TrajectoryBuffer(max_size=1000)
trainer   = PPOTrainer(model, tokenizer, optimizer, buffer)
dataset   = MBPPDataset("google-research/mbpp/mbpp_train.jsonl")

print(f"✅ PPOTrainer init · params={sum(p.numel() for p in model.parameters()):,}")

# 2. Sampling function
def rollout_to_buffer(rollouts, batch_size):
    buffer.clear()
    for _ in range(rollouts):
        batch     = [dataset.sample() for _ in range(batch_size)]
        prompts   = [p for p, _ in batch]
        gts       = [c for _, c in batch]

        gen, _, _, lens = trainer.rollout(prompts)
        lp  = trainer.compute_log_probs(prompts, gen, lens)
        if torch.isnan(lp).any(): continue

        rewards = [reward_function(g, gt) for g, gt in zip(gen, gts)]
        if not all(np.isfinite(r) for r in rewards): continue

        for pr, g, r, l in zip(prompts, gen, rewards, lp.tolist()):
            buffer.add(pr, g, float(r), float(l))

# 3. PPO training function
def train_ppo(cfg):

    best_reward, no_improve = -1e9, 0
    for epoch in range(1000):
        rollout_to_buffer(rollouts=12, batch_size=cfg.batch_size)

        if len(buffer) >= cfg.batch_size:
            loss = trainer.update(buffer, batch_size=cfg.batch_size)
            buffer.clear()
            if loss is None or torch.isnan(torch.tensor(loss)): continue

        if (epoch + 1) % cfg.eval_interval == 0:
            eval_prompts = [dataset.sample()[0] for _ in range(16)]
            gen_eval, _  = trainer.rollout(eval_prompts)
            gt_eval      = [dataset.lookup_ground_truth(p) for p in eval_prompts]
            rewards_eval = [reward_function(g, gt) for g, gt in zip(gen_eval, gt_eval)]
            avg_reward   = float(np.mean(rewards_eval))

            print(f"[{epoch+1}] avg_reward={avg_reward:.4f}")
            wandb.log({"epoch": epoch+1, "avg_reward": avg_reward, "loss": loss or 0.0})

            if avg_reward > best_reward:
                best_reward, no_improve = avg_reward, 0
                save_dir = f"saved_nanoGPT_finetuned/PPO_best_step_{epoch+1}"
                model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
                print(f"💾 Improved, saved to {save_dir}")
            else:
                no_improve += 1
                print(f"⚠️ No improvement for {no_improve} evals")

            if no_improve >= cfg.early_stop_patience:
                print("🛑 Early stopping"); break

# 4. Entrance
if __name__ == "__main__":
    if not wandb.run:
        wandb.init(project="nanoGPT-RL-PPO",
                   config=dict(lr=1e-5, batch_size=8, max_new_tokens=100,
                               early_stop_patience=500, eval_interval=100,
                               ppo_epochs=4, log_interval=10))
    cfg = wandb.config
    train_ppo(cfg)
