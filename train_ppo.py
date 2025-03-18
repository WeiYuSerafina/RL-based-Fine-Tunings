from transformers import AutoModelForCausalLM, AutoTokenizer
from trajectory_buffer import TrajectoryBuffer
from ppo_trainer import PPOTrainer

# Load model nanoGPT-RL
model_name = "kliu128/nanoGPT-RL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.pretrained(model_name)

# Create PPO Trainer
buffer = TrajectoryBuffer()
trainer = PPOTrainer(model, tokenizer, buffer)

# Train loops
for epoch in range(10):
    loss = trainer.train_step(batch_size = 32)
    print(f"Epoch {epoch + 1}, Loss: {loss}")



