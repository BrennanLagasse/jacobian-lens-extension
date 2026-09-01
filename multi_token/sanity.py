import torch
import torch.nn.functional as F

data_path = "./results/phrase_means.pt"

data = torch.load(data_path, map_location="cpu")

new_phrases = list(data['phrase_means'].keys())

# Normalize (L2) and stack embeddings
new_phrase_weights = torch.stack(
    [F.normalize(v, p=2, dim=0) for v in data['phrase_means'].values()], 
    dim=0
)

print(new_phrase_weights)

print(torch.norm(new_phrase_weights, dim=1))