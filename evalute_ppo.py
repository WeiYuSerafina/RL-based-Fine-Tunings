import torch
from nano_gpt_policy import NanoGPTPolicy
from transformers import AutoTokenizer

# 1. 重新加载模型（更好）
model_name = "kliu128/nanoGPT-RL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = NanoGPTPolicy(...)  # 或 AutoModelForCausalLM.from_pretrained(model_name)

# 加载你训练好的权重（如果保存了）
# model.load_state_dict(torch.load("path/to/your_trained_model.pt"))

model.eval()  # Important! Set model to eval mode


def evaluate_model(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)  # 保证数据和模型在同一个设备上

    with torch.no_grad():  # 不计算梯度，加速
        output = model.generate(input_ids, max_length=100)

    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_code


# 测试
prompt = "Write a function to compute factorial using recursion."
generated_code = evaluate_model(model, tokenizer, prompt)
print(generated_code)
