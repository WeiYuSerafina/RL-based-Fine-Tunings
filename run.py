import torch
import os
import json
import time
import sys
from datetime import datetime
from nano_gpt_policy import NanoGPTPolicy
from ppo_trainer import PPOTrainer
from trajectory_buffer import TrajectoryBuffer
from reward_function import reward_function
from dataset_loader import ArcadeDataset

# 1. Load model
model_name = "./saved_nanoGPT"
model = NanoGPTPolicy(model_name)

# 2. Optimizer and buffer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
buffer = TrajectoryBuffer()
ppo = PPOTrainer(model, optimizer, buffer)

# 3. Load dataset
dataset_path = 'arcade-nl2code/arcade_nl2code/annotated_dataset/converted_new_tasks.jsonl'
dataset = ArcadeDataset(dataset_path)

# 4. Training loop
for step in range(1000):  # Total steps
    # ---- Sample prompt and ground truth ----
    prompt, ground_truth = dataset.sample()

    # ---- Encode prompt ----
    # input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids

    # ---- Generate output (同时拿到log_probs) ----
    output_ids, all_log_probs = model.generate(input_ids, max_new_tokens=100)
    new_tokens = output_ids[:, input_ids.shape[1]:]  # 保留所有 batch 中的新 token
    generated_code = model.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    # ---- Debug ructionruction...----
    if "ructionruction" in generated_code:
        print("Detected repetition artifact in output.")
        reward = 0.0  # Optional: You can also continue to skip this sample
    else:
        # ---- Compute reward ----
        reward = reward_function(generated_code, ground_truth, prompt=prompt)

    # ---- Compute average log_prob ----
    avg_log_prob = all_log_probs.mean().item()

    # ---- Store transition into buffer ----
    buffer.add(prompt, generated_code, reward, avg_log_prob)

    # ---- PPO Update ----
    if step > 0 and len(buffer) > 0:
        ppo.update(buffer)  # 自动适配 batch_size
        buffer.clear()

    # ---- Logging ----
    # if step % 100 == 0:
    #    print(f"Step {step}: Reward = {reward:.4f}, Avg Log Prob = {avg_log_prob:.4f}")

    # ---- Debug/Logging enhance ----
    if step % 50 == 0:
        print(f"Step {step} Debug Prompt:\n{prompt}")
        print(f"Tokenized length: {input_ids.shape[1]}\n")

    if step % 100 == 0:
        print(f"Step {step}:")
        print("Full prompt sent to model:\n", prompt)
        print(f"Generated: {generated_code}") # print(f"Generated: {generated_code[:60]}")
        print(f"Reward = {reward:.4f}, Avg Log Prob = {avg_log_prob:.4f}")

# 5. Save fine-tuned model
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"./saved_nanoGPT_finetuned/PPO/{timestamp}"
os.makedirs(save_path, exist_ok=True)

# save pytorch_model.bin
torch.save(model.model.state_dict(), f"{save_path}/pytorch_model.bin")

# save config.json
with open(f"{save_path}/config.json", "w") as f:
    json.dump(model.model.config.__dict__, f, indent=4)

# save tokenizer（if have）
if model.tokenizer:
    model.tokenizer.save_pretrained(save_path)

# Save the standard ppo_model.pt file for use in evaluate_perplexity.py
torch.save(model.model.state_dict(), f"{save_path}/ppo_model.pt")

print(f"Fine-tuned PPO model and related files saved to: {save_path} (includes pytorch_model.bin, config.json,tokenizer, ppo_model.pt)")

# 6.create logs file
log_dir = "logs/logs PPO"
os.makedirs(log_dir, exist_ok=True)

# Set the log file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"ppo_run_{timestamp}.log")

# Redirect stdout to file + console
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = TeeLogger(log_file_path)
