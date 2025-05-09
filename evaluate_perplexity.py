import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from model import GPT, GPTConfig  # It depends on the actual path

# Load baseline model (trained and saved via torch.save(model.state_dict(), ...))
state_dict = torch.load("saved_nanoGPT/baseline_model.pt", map_location="cpu")
print("baseline_model.pt type:", type(state_dict))
print("First 5 parameter names:", list(state_dict.keys())[:5])

# -------- Custom Perplexity Calculation Function --------
def compute_perplexity(model, tokenizer, texts, device='cpu', stride=128):
    model.eval()
    model.to(device)

    max_length = model.config.block_size  # block_size = 128
    nll_sum = 0.0
    n_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        seq_len = input_ids.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - begin_loc
            input_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_slice.clone()
            target_ids[:, :-trg_len] = -100  # Only calculate the loss of the current window

            with torch.no_grad():
                logits, neg_log_likelihood = model(input_slice, targets=target_ids)

            nll_sum += neg_log_likelihood.item() * trg_len
            n_tokens += trg_len

            if end_loc == seq_len:
                break

    avg_nll = nll_sum / n_tokens
    perplexity = torch.exp(torch.tensor(avg_nll))
    return perplexity.item()

# -------- Main program entry --------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialization tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Model configuration (based on the GPTConfig settings you used for training)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
    )

    # The text to be evaluated (you can use the validation set samples or your own)
    texts = [
        "Write a Python function to reverse a list.",
        "Sort a list of integers using bubble sort.",
        "Print numbers from 1 to 10 using a for loop.",
    ]

    # Load baseline model
    model = GPT(config)
    state_dict_base = torch.load("saved_nanoGPT/baseline_model.pt", map_location=device)
    if any(k.startswith("_orig_mod.") for k in state_dict_base.keys()):
        state_dict_base = {k.replace("_orig_mod.", ""): v for k, v in state_dict_base.items()}
        print("Cleaned '_orig_mod.' prefix in baseline weights")

    missing, unexpected = model.load_state_dict(state_dict_base, strict=False)
    print("Baseline missing keys:", missing)
    print("Baseline unexpected keys:", unexpected)

    ppl_base = compute_perplexity(model, tokenizer, texts, device)
    print(f"[Baseline] Perplexity: {ppl_base:.2f}")

    # Load PPO model
    state_dict_ppo = torch.load("saved_nanoGPT_finetuned/PPO/2025-04-24_23-04-30/pytorch_model.bin",
                                map_location=device)
    state_dict_ppo = {k.replace("_orig_mod.", ""): v for k, v in state_dict_ppo.items()}

    missing, unexpected = model.load_state_dict(state_dict_ppo, strict=False)
    print("PPO missing keys:", missing)
    print("PPO unexpected keys:", unexpected)

    ppl_ppo = compute_perplexity(model, tokenizer, texts, device)
    print(f"[PPO] Perplexity: {ppl_ppo:.2f}")

    # Load A2C model
    state_dict_a2c = torch.load("saved_nanoGPT_finetuned/A2C/2025-04-30_18-38-52/pytorch_model.bin",
                                map_location=device)
    state_dict_a2c = {k.replace("_orig_mod.", ""): v for k, v in state_dict_a2c.items()}

    missing, unexpected = model.load_state_dict(state_dict_a2c, strict=False)
    print("A2C missing keys:", missing)
    print("A2C unexpected keys:", unexpected)

    ppl_a2c = compute_perplexity(model, tokenizer, texts, device)
    print(f"[A2C] Perplexity: {ppl_a2c:.2f}")

