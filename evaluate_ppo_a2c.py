import torch
import json
from nano_gpt_ppo_policy import NanoGPTPolicy  # 自定义模型类

import json

def load_prompt_completion_pairs(path: str, max_samples: int = 1000):
    # 1. 读文件（jsonl 或 json）
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            # 直接构成列表，后面可 len()、slice()
            records = [json.loads(line) for line in f]
    else:  # .json
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)                     # list[dict]

    # 2. 提取 prompt + completion/code
    pairs = []
    for obj in records:
        prompt = obj.get("prompt", "").strip()

        # MBPP train/valid 用 "completion"，sanitized 用 "code"
        completion = (obj.get("completion")             # 优先取 completion
                      or obj.get("code", "")).strip()    # 否则取 code

        if prompt and completion:
            pairs.append((prompt, f"{prompt} {completion}"))
            if len(pairs) >= max_samples:
                break

    return pairs


def evaluate_perplexity(model, tokenizer, prompt_full_pairs, batch_size=8, max_length=256):
    model.eval()
    device = model.device
    total_loss = 0
    total_tokens = 0

    for start_idx in range(0, len(prompt_full_pairs), batch_size):
        batch_pairs = prompt_full_pairs[start_idx:start_idx + batch_size]
        prompts, fulls = zip(*batch_pairs)

        encodings = tokenizer(list(fulls), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        with torch.no_grad():
            labels = input_ids.clone()
            for i, (prompt, full) in enumerate(batch_pairs):
                prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
                prompt_len = len(prompt_ids)
                prompt_len = min(prompt_len, input_ids.size(1))  # 防止越界
                labels[i, :prompt_len] = -100  # 屏蔽 prompt 区域

                valid_token_count = (labels[i] != -100).sum().item()
                if valid_token_count < 5:
                    print(f"⚠️ Skipping sample {i + start_idx} with only {valid_token_count} valid tokens (prompt too long?)")
                    labels[i] = -100  # 忽略整条

                full_len = attention_mask[i].sum().item()
                print(f"[Check] Prompt len: {prompt_len}, Full len: {full_len}")

            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            # 插入：打印高损失样本
            per_token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view_as(labels)

            for i in range(len(batch_pairs)):
                loss_i = per_token_loss[i][labels[i] != -100].mean().item()
                if loss_i > 1:  # 可根据情况调整阈值
                    prompt, full = batch_pairs[i]
                    completion = full[len(prompt):].strip()
                    print(f"\n🚨 High-loss Sample {i + start_idx}:")
                    print(f"📌 Prompt:\n{prompt}")
                    print(f"📌 Completion:\n{completion}")
                    print(f"📉 Completion Loss: {loss_i:.4f}")

        if (start_idx // batch_size + 1) % 20 == 0:
            print(f"🔄 Batch {start_idx // batch_size + 1}: Running Avg Loss = {total_loss / total_tokens:.4f}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"✅ Evaluation Complete. Final Avg Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")
    return perplexity.item()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to PPO or A2C fine-tuned model folder")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to validation .jsonl file")
    parser.add_argument("--max_samples", type=int, default=1000, help="Number of prompts to evaluate")
    args = parser.parse_args()

    # 1. Load model and tokenizer
    model = NanoGPTPolicy(args.model_path,
                          tokenizer_path="./data/mbpp_new")  # ← 显式给路径
    tokenizer = model.tokenizer

    # 2. Load prompt + completion pairs
    pairs = load_prompt_completion_pairs(args.jsonl_path, max_samples=args.max_samples)

    # 3. Evaluate
    print(f"📊 Evaluating PPO/A2C model on {len(pairs)} prompts...")
    ppl = evaluate_perplexity(model, tokenizer, pairs)
    print(f"✅ Final Perplexity: {ppl:.2f}")

    """
python evaluate_ppo_a2c.py \
  --model_path saved_nanoGPT_finetuned/PPO_best_step_0 \
  --jsonl_path google-research/mbpp/sanitized-mbpp.json \
  --max_samples 500
    """
