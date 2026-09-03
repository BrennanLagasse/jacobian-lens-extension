import torch

hbar_path = "./results/h_bar.pt"

data = torch.load(hbar_path, map_location="cpu")

print(data)
print(data.shape)