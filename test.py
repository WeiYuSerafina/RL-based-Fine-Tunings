import torch
ckpt = torch.load("/Users/serafinayu/PycharmProjects/nanoGPT-RL/out/mbpp_baseline_v2/ckpt.pt", map_location="cpu")
print(ckpt.keys())          # 打印顶层键名
