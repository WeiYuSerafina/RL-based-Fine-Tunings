import torch
import os
import json
import sys
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
from a2c_trainer import A2CTrainer
from trajectory_buffer_a2c import TrajectoryBuffer
from reward_function import reward_function
from train_a2c import train
from nano_gpt_a2c_policy import NanoGPTA2CPolicy

# Redirect stdout to log file + console
def setup_logger():
    log_dir = "logs/logs_A2C"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"a2c_run_{timestamp}.log")

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
    return timestamp  # return for model save path

def main():
    # 0. Setup logging
    timestamp = setup_logger()

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model using policy wrapper
    model_name = "./saved_nanoGPT"
    model = NanoGPTA2CPolicy(model_name).to(device)

    # 3. Tokenizer
    tokenizer = model.tokenizer or AutoTokenizer.from_pretrained("gpt2")
    # Set pad_token（GPT2 does not have）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Dataset
    dataset = load_dataset("json", data_files="arcade-nl2code/arcade_nl2code/annotated_dataset/new_tasks_for_nanoGPT.jsonl")["train"]

    # 5. Buffer and trainer
    buffer = TrajectoryBuffer()
    trainer = A2CTrainer(model, buffer, reward_fn=reward_function, device=device, pad_token_id=tokenizer.pad_token_id )

    # 6. Train
    train(model, dataset, trainer, tokenizer, device)

    # 7. Save trained model
    save_path = f"./saved_nanoGPT_finetuned/A2C/{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.model.state_dict(), f"{save_path}/pytorch_model.bin")

    with open(f"{save_path}/config.json", "w") as f:
        json.dump(model.model.config.__dict__, f, indent=4)

    if model.tokenizer:
        model.tokenizer.save_pretrained(save_path)

    print(f"✅ Fine-tuned model, config, and tokenizer saved to: {save_path}")

if __name__ == "__main__":
    main()
