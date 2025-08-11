from __future__ import annotations

import os
import sys
import csv
import json
import math
import random
from datetime import datetime
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any

import torch
import wandb
from datasets import load_dataset

from a2c_trainer import A2CTrainer
from trajectory_buffer_a2c import TrajectoryBuffer
from reward_function import reward_function
from nano_gpt_a2c_policy import NanoGPTA2CPolicy
from evaluate_ppo_a2c_perplexity import evaluate_perplexity, load_prompt_completion_pairs

# logging utils
def log_ppl_to_csv(step: int, ppl: float, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "ppl"])
        w.writerow([step, ppl])

def setup_logger() -> str:
    log_dir = "logs/logs_A2C"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"a2c_run_{timestamp}.log")

    class Tee:
        def __init__(self, fp):
            self.terminal = sys.stdout
            self.file = open(fp, "w", encoding="utf-8")

        def write(self, msg):
            self.terminal.write(msg)
            self.file.write(msg)

        def flush(self):
            self.terminal.flush()
            self.file.flush()

    sys.stdout = Tee(log_file)
    return timestamp

# sweep config
sweep_config = {
    "method": "random",
    "metric": {"name": "moving_avg_reward", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 1e-6, "max": 5e-5},
        "batch_size": {"values": [8, 16, 32]},
        "total_steps": {"value": 4000},
        "eval_interval": {"values": [100, 200]},
        "early_stop_patience": {"values": [200, 300]},
    },
}

# wandb-safe scalar
def _scalar(x: Any):
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    try:
        x = float(x)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x

def _rollout_batch(
    *,
    model: NanoGPTA2CPolicy,
    tokenizer,
    dataset,
    buffer: TrajectoryBuffer,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    debug: bool = False,
) -> None:

    model.eval()
    pad_id = tokenizer.pad_token_id
    block_size = getattr(model.model.config, "block_size", None)

    # Random sampling
    idxs = random.sample(range(len(dataset)), k=min(batch_size, len(dataset)))

    for i in idxs:
        ex = dataset[i]
        prompt = ex.get("prompt", ex.get("text", ""))

        # ground-truth供 reward
        reference_code = ex.get("code") or ex.get("completion") or ""
        if isinstance(reference_code, list):
            reference_code = reference_code[0]

        # encode prompt (no special tokens)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)  # [1, Tp]

        # block_size truncates prompt
        if block_size is not None and input_ids.size(1) > block_size:
            input_ids = input_ids[:, :block_size]

        # baseline value
        with torch.no_grad():
            out0 = model(input_ids=input_ids, attention_mask=None, labels=None)
            value0 = out0["value"]  # [1,1]

        # autoregressive generate
        generated_ids = input_ids[0].clone()  # [Tp]
        gen_actions = []
        gen_logps = []

        for _ in range(max_new_tokens):
            cur_in = generated_ids.unsqueeze(0)
            if block_size is not None and cur_in.size(1) > block_size:
                cur_in = cur_in[:, -block_size:]

            with torch.no_grad():
                out = model(input_ids=cur_in, attention_mask=None, labels=None)
                logits = out["logits"]  # [1, cur_T, V]

            probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze(0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()                # scalar
            logp = dist.log_prob(action)

            gen_actions.append(action)
            gen_logps.append(logp)

            generated_ids = torch.cat([generated_ids, action.unsqueeze(0)], dim=0)

            # stop when EOS
            if action.item() == tokenizer.eos_token_id:
                break

            # hard safety: stop if > block_size after append
            if block_size is not None and generated_ids.numel() >= block_size:
                break

        # decode full output
        decoded_output = tokenizer.decode(generated_ids.tolist())

        # reward
        reward = reward_function(
            generated_code=decoded_output,              # named args for clarity
            reference_code=reference_code,
            prompt=prompt,
        )
        done = True

        # average log prob (diagnostic)
        if gen_logps:
            avg_logp = torch.stack(gen_logps).mean()
        else:
            avg_logp = torch.tensor(0.0, device=device)

        # push to buffer
        # states = full sequence (prompt+gen)
        # actions = generated token seq (List[tokens])
        # log_probs = per-token logp seq
        full_state_cpu = generated_ids.cpu()
        if gen_actions:
            actions_cpu = torch.stack(gen_actions).cpu()
            logps_cpu = torch.stack(gen_logps).cpu()
        else:
            actions_cpu = torch.tensor([tokenizer.eos_token_id])
            logps_cpu = torch.tensor([0.0])

        buffer.store(
            full_state_cpu,
            actions_cpu,
            reward,
            done,
            logps_cpu,
            value0.squeeze(0).cpu(),
        )

        if debug and i == idxs[0]:  # print 1 example per batch for brevity
            print(
                f"[ROLLOUT] reward={reward:.4f} len_gen={len(gen_actions)} "
                f"avg_logp={avg_logp.item():.4f}"
            )

    model.train()  # leave model back in train mode for optimizer step

# main training loop
def run(cfg: SimpleNamespace):
    timestamp = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Device: {device} ===")

    # Generate a unified save root directory
    save_root = f"./saved_nanoGPT_finetuned/A2C/{timestamp}"
    os.makedirs(save_root, exist_ok=True)

    # load model
    model_dir = "./out/mbpp_baseline_v3/"
    tok_dir = "./data/mbpp_new"
    model = NanoGPTA2CPolicy(model_dir, tokenizer_path=tok_dir, debug=getattr(cfg, "debug", False))
    model.to(device)  # move full policy (backbone+value)
    tokenizer = model.tokenizer

    # sanity
    print("=== Model/Tokenizer Sanity ===")
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {num_params:,}; trainable: {trainable_params:,}")
    print(f"Tokenizer vocab size: {len(tokenizer)} | pad_token_id={tokenizer.pad_token_id}")
    if hasattr(model.model.config, "vocab_size"):
        print(f"Model config vocab_size: {model.model.config.vocab_size}")
    block_size = getattr(model.model.config, "block_size", None)
    if block_size is not None:
        print(f"Model block_size: {block_size}")

    # debug for a2c_trainer.py pad_token_id: int = 0
    print("DEBUG ‑ pad_token_id =", tokenizer.pad_token_id)

    # dataset
    raw_ds = load_dataset("json", data_files="google-research/mbpp/mbpp_train.jsonl")
    train_ds = raw_ds["train"]

    # trainer
    buffer = TrajectoryBuffer()
    trainer = A2CTrainer(
        model,
        buffer,
        reward_fn=reward_function,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        debug=getattr(cfg, "debug", False),
    )

    debug = getattr(cfg, "debug", False)

    # trackers
    best_ppl = float("inf")  # ← NEW
    best_state = None
    best_step = 0
    patience_counter = 0
    reward_window: List[float] = []

    # training steps: rollout → update
    for step in range(cfg.total_steps):

        # collect experience
        _rollout_batch(
            model=model,
            tokenizer=tokenizer,
            dataset=train_ds,
            buffer=buffer,
            device=device,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
            debug=debug and (step % 10 == 0),
        )

        # optimize
        metrics: Dict[str, Any] = trainer.train_step()

        if debug or step < 3:
            print(f"[DEBUG] raw metrics @step{step}: {metrics}")

        # reward proxy
        reward = float(metrics.get("return_mean", 0.0) or 0.0)

        # moving avg (50)
        reward_window.append(reward)
        if len(reward_window) > 50:
            reward_window.pop(0)
        moving_avg = sum(reward_window) / len(reward_window)

        # wandb metric filtering
        wanted_keys = (
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "adv_mean",
            "return_mean",
            "lm_loss",
        )
        key_metrics = {k: _scalar(metrics.get(k, None)) for k in wanted_keys}
        key_metrics = {k: v for k, v in key_metrics.items() if v is not None}

        if debug or step < 3:
            print(f"[DEBUG] wandb key_metrics @step{step}: {key_metrics}")

        wandb.log(
            {
                "reward": reward,
                "moving_avg_reward": moving_avg,
                **key_metrics,
            },
            step=step,
        )

        # save every N steps of checkpoint
        if (step + 1) % 100 == 0:
            ckpt_step_path = os.path.join(save_root, f"ckpt_{step + 1}.pt")
            torch.save(model.state_dict(), ckpt_step_path)

        # eval ppl
        if step % cfg.eval_interval == 0:
            # build eval subset
            eval_pairs = load_prompt_completion_pairs(
                path="google-research/mbpp/sanitized-mbpp.json",
                max_samples=50
            )

            ppl_val = evaluate_perplexity(
                model=model,
                tokenizer=tokenizer,
                prompt_full_pairs=eval_pairs,
                batch_size=8,
                max_length=256
            )

            log_ppl_to_csv(step, ppl_val, "logs/ppl_a2c.csv")
            wandb.log({"ppl": ppl_val}, step=step)
            print(f"[Step {step}] PPL(A2C) = {ppl_val:.2f} | AvgR = {moving_avg:.4f}")

            # To verify PPL, select the best ckpt and do early-stop
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_step = step
                patience_counter = 0        # reset patience
            else:
                patience_counter += 1

            if patience_counter >= cfg.early_stop_patience:
                print(f"Early stopped at step {step} (best_step={best_step}).")
                break

    # save checkpoints
    final_dir = f"./saved_nanoGPT_finetuned/A2C/{timestamp}"
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))

    try:
        cfg_dict = model.model.config.to_dict()
    except Exception:
        cfg_dict = model.model.config.__dict__
    with open(os.path.join(final_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=4, default=str)

    tokenizer.save_pretrained(final_dir)

    best_dir = f"./saved_nanoGPT_finetuned/A2C_best_step_{best_step}"
    os.makedirs(best_dir, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, os.path.join(best_dir, "pytorch_model.bin"))
    with open(os.path.join(best_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=4, default=str)
    tokenizer.save_pretrained(best_dir)

    print(
        f"\n✅ Training complete. Final ckpt → {final_dir}  |  "
        f"Best ckpt(step {best_step}) → {best_dir} (Best PPL={best_ppl:.4f})"
    )

# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run WandB sweep agent")
    parser.add_argument("--debug", action="store_true", help="Verbose debug prints")
    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="nanoGPT-RL-A2C")
        wandb.agent(sweep_id, function=lambda: run(SimpleNamespace(**wandb.config)))
    else:
        default_cfg = SimpleNamespace(
            lr=3e-5,
            batch_size=16,
            total_steps=2000,
            eval_interval=50,
            early_stop_patience=6, # based on eval_interval（= 6*50 = 300 step）
            max_new_tokens=100,
            num_epochs=4,            # currently unused in main loop; kept for compat
            eval_max_length=256,
            debug=False
        )
        wandb.init(project="nanoGPT-RL-A2C", config=default_cfg.__dict__)
        run(default_cfg)
