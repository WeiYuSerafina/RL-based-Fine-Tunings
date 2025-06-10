# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
import time
out_dir = 'out-shakespeare-gpt2'   # 模型训练输出的目录，自定义一个不和原始 data 冲突
dataset = 'shakespeare'            # 表示使用的是 data/shakespeare 目录下的数据

wandb_log = False
wandb_project = 'nanoGPT-RL_baseline_debug'
wandb_run_name = f'baseline_run_{time.time()}'

init_from = 'scratch'  # ← 不加载任何旧 checkpoint

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 8
block_size = 256
gradient_accumulation_steps = 2

# this makes total number of tokens be 300B
max_iters = 1000
lr_decay_iters = 1000 # should be ~= max_iters per Chinchilla

# eval stuff
eval_interval = 100
eval_iters = 20
log_interval = 10

# weight decay
weight_decay = 1e-2
