import torch
import math
import json
from tqdm import tqdm
from transformers import GPT2Tokenizer
from model import GPT, GPTConfig  # Adjust if needed

# -------- Custom Perplexity Calculation Function --------
def compute_perplexity(model, tokenizer, texts, device='cpu', stride=128):
    model.eval()
    model.to(device)

    max_length = model.config.block_size
    total_loss = 0.0
    total_tokens = 0

    for text_id, text in enumerate(texts):
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        seq_len = input_ids.size(1)
        print(f"\n[Text {text_id+1}] Length: {seq_len} tokens")

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            input_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_slice.clone()
            target_ids[:, 0] = -100  # ignore first token in each slice

            with torch.no_grad():
                logits, loss = model(input_slice, targets=target_ids) # prove forward() in model.py return logits, loss

            if text_id == 0 and begin_loc == 0:  # 只打印第一个样本的第一段
                print("\n🔍 Sample Prompt:")
                print(text)

                # logits shape: [batch, seq_len, vocab_size]
                first_token_logits = logits[0, 0]  # 第一个位置的 logits
                probs = torch.nn.functional.softmax(first_token_logits, dim=-1)
                topk = torch.topk(probs, k=10)

                print("\n🧠 Top-10 Predicted Tokens at Position 0:")
                for i in range(10):
                    token_id = topk.indices[i].item()
                    prob = topk.values[i].item()
                    token = tokenizer.decode([token_id])
                    print(f"{i + 1:>2}. '{token}': {prob:.4f}")

            loss_val = loss.item()
            total_loss += loss_val * (target_ids != -100).sum().item()
            total_tokens += (target_ids != -100).sum().item()

            print(f"Segment [{begin_loc}:{end_loc}] loss = {loss_val:.4f}")

            if end_loc == seq_len:
                break

    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)
    print(f"Tokens: {total_tokens}, Total Loss: {total_loss:.4f}, Avg NLL: {avg_nll:.4f}, Perplexity: {perplexity:.2f}")
    return perplexity

# -------- Main program entry --------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize tokenizer (must match training)
    tokenizer = GPT2Tokenizer.from_pretrained("data/arcade_new", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # 🔍 DEBUG: confirm tokenizer status
    print("✅ Successfully loaded  tokenizer:", type(tokenizer))
    print("🧩 pad_token:", tokenizer.pad_token)
    print("🧠 vocab size:", tokenizer.vocab_size)

    # Model configuration (must match training)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        n_layer=2,
        n_head=2,
        n_embd=128,
        bias=True,
    )

    # Sample test texts
    def load_test_prompts(jsonl_path, n=500):
        prompts = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                task = json.loads(line)

                instruction = task.get("instruction", "").strip()
                context = task.get("context", "").strip()
                solution = task.get("solution", "").strip()

                if instruction and solution:  # context 可为空
                    # 注意：GPT2 tokenizer 会自动处理 <|endoftext|>
                    full_prompt = f"Instruction: {instruction}\nContext: {context}\n{solution}"
                    prompts.append(full_prompt)

        return prompts


    # Extract 500 samples
    test_prompts = load_test_prompts("/Users/serafinayu/PycharmProjects/nanoGPT-RL/arcade-nl2code/arcade_nl2code/annotated_dataset/converted_new_tasks.jsonl", n=500)

    # Load baseline model
    # Load config from JSON
    with open("saved_nanoGPT/config_v4_debug.json", "r") as f:
        config_dict = json.load(f)
    config = GPTConfig(**config_dict)

    # Load model
    print("Loading baseline_model_v4_debug.pt...")
    model = GPT(config)
    state_dict = torch.load("saved_nanoGPT/baseline_model_v4_debug.pt", map_location=device)

    # Clean DDP prefixes if any
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print("Cleaned '_orig_mod.' prefix in baseline weights")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)} - {missing}")
    print(f"Unexpected keys: {len(unexpected)} - {unexpected}")

    # Evaluate perplexity
    ppl = compute_perplexity(model, tokenizer, test_prompts, device)
    print(f"\n[Baseline] Perplexity: {ppl:.4f}")

    """
    # Load PPO model
   
    state_dict_ppo = torch.load("saved_nanoGPT_finetuned/PPO/2025-05-14_11-48-56/pytorch_model.bin",
                                map_location=device)
    state_dict_ppo = {k.replace("_orig_mod.", ""): v for k, v in state_dict_ppo.items()}

    missing, unexpected = model.load_state_dict(state_dict_ppo, strict=False)
    print("PPO missing keys:", missing)
    print("PPO unexpected keys:", unexpected)

    ppl_ppo = compute_perplexity(model, tokenizer, texts, device)
    print(f"[PPO] Perplexity: {ppl_ppo:.2f}")

    # Load A2C model
    state_dict_a2c = torch.load("saved_nanoGPT_finetuned/A2C/2025-05-14_12-12-57/pytorch_model.bin",
                                map_location=device)
    state_dict_a2c = {k.replace("_orig_mod.", ""): v for k, v in state_dict_a2c.items()}

    missing, unexpected = model.load_state_dict(state_dict_a2c, strict=False)
    print("A2C missing keys:", missing)
    print("A2C unexpected keys:", unexpected)

    ppl_a2c = compute_perplexity(model, tokenizer, texts, device)
    print(f"[A2C] Perplexity: {ppl_a2c:.2f}")
    """
