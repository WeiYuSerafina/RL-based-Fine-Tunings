from train_ppo import model, tokenizer

def evaluate_model(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors = "pt")
    output = model.generate(input_ids, max_length=100)
    generated_code = tokenizer.decode(output[0], skip_special_tokens = True)
    return generated_code

# Test codes after the PPO training
prompt = "Write a function to compute factorial using recursion."
generated_code = evaluate_model(model, tokenizer, prompt)
print(generated_code)