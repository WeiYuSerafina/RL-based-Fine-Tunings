from transformers import AutoModelForCausalLM, AutoTokenizer
from trajectory_buffer import TrajectoryBuffer
from ppo_trainer import PPOTrainer
from reward_function import reward_function
from dataset_loader import ArcadeDataset  # 你新写的小类

# Load model
model_name = "kliu128/nanoGPT-RL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create PPO Trainer
buffer = TrajectoryBuffer(max_size=500)
trainer = PPOTrainer(model, tokenizer, buffer)

# Load your new_tasks_for_nanoGPT.jsonl
dataset = ArcadeDataset("arcade-nl2code/arcade_nl2code/annotated_dataset/new_tasks_for_nanoGPT.jsonl")

# Fill buffer
for _ in range(100):
    prompt, ground_truth = dataset.sample()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=50)
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    reward = reward_function(generated_code, ground_truth)

    buffer.store(
        state=prompt,
        action=generated_code,
        reward=reward,
        next_state=prompt  # 可以先这样
    )

print(f"Buffer填充完成，当前存储数量：{len(buffer.states)}")

# Train loop
for epoch in range(10):
    print(f"🔁 Epoch {epoch + 1}")
    loss = trainer.update(buffer, batch_size=8)
    print(f"✅ Epoch {epoch + 1} Finish, Loss: {loss}")

