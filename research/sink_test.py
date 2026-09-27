import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
M="Qwen/Qwen2.5-1.5B"
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
WU=model.get_output_embeddings().weight.detach()
texts=["Connecticut is a state in the northeastern United States.","Cheating on an exam can lead to expulsion.","The recipe calls for two cups of flour and a pinch of salt.","Interest rates were left unchanged at the central bank's meeting."]
with torch.no_grad():
    H=[model.model(**tok(t,return_tensors="pt"),use_cache=False).last_hidden_state[0] for t in texts]
for t,h in zip(texts,H):
    print(f"{t[:30]:<30} post-norm ||h|| by position:", [round(x,1) for x in h.norm(dim=1)[:6].tolist()])
sink=F.normalize(torch.stack([h[0] for h in H]).mean(0),dim=0)
gen=torch.cat([h[4:] for h in H])
print("cos(sink dir, generic h): mean %.3f" % F.cosine_similarity(gen,sink[None],dim=1).mean())
print("logit of unit sink-dir row on generic h: mean %.1f (real max logits ~21-23)" % (gen@sink).mean())
# a 150-sample mean with k position-0 samples mixed in: how much does the sink dominate?
typ=torch.stack([h[5] for h in H]).mean(0)
for k in (0,5,15):
    mix=(k*H[0][0]+(150-k)*typ)/150
    print(f"k={k:>2}/150 position-0 samples in mean -> cos(mean, sink dir) = {F.cosine_similarity(mix[None],sink[None]).item():.3f}")
