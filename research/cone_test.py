"""Toy replication of Brennan's PRIOR_REPRESENTATION_EMBED_PROJ recipe on Qwen2.5-0.5B (CPU).

Question: does the centered, unit-norm prototype row for a phrase produce logits that
dominate the real vocabulary on *unrelated* prompts (i.e. is it a prompt-independent attractor)?
"""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
M = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
WU = model.get_output_embeddings().weight.detach()  # (V, d)

PHRASE = "New York"
ctx = [
    "After graduating she moved to New York to work in publishing.",
    "The flight from Chicago to New York was delayed by two hours.",
    "He grew up in a small town but always dreamed of living in New York.",
    "The company opened its first office in New York in 1998.",
    "Millions of tourists visit New York every year.",
    "She took the train down to New York for the weekend.",
    "The museum in New York holds one of the largest collections in the world.",
    "Rents in New York have risen sharply over the past decade.",
    "Our headquarters are located in New York and London.",
    "The marathon in New York attracts runners from around the globe.",
    "He was born in New York and raised in New Jersey.",
    "The subway system in New York runs twenty-four hours a day.",
    "They celebrated the holiday in New York with friends.",
    "The stock exchange in New York opened lower on Monday.",
    "A blizzard hit New York and shut down the airports.",
    "The mayor of New York announced a new housing plan.",
    "She studied art history at a university in New York.",
    "Pizza in New York is famous for its thin, foldable slices.",
    "The parade through New York drew huge crowds.",
    "The band played their final show in New York last night.",
]
generic = [
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The committee will meet on Thursday to review the budget.",
    "He tightened the bolts and checked the oil before starting the engine.",
    "The novel follows three generations of a farming family.",
    "Interest rates were left unchanged at the central bank's meeting.",
    "The children built a sandcastle and watched the tide come in.",
    "The software update fixes several security vulnerabilities.",
    "Volcanic soil is unusually rich in minerals.",
    "She practiced the violin for an hour every morning.",
    "The referee blew the whistle to end the first half.",
    "Ancient traders carried silk and spices along the route.",
    "The patient was advised to rest and drink plenty of fluids.",
    "Quarterly earnings beat analysts' expectations.",
    "The telescope captured images of a distant galaxy.",
    "The bakery on the corner sells sourdough on weekends.",
    "The treaty was signed after months of negotiation.",
    "Bees communicate the location of flowers through dance.",
    "The bridge was closed for repairs over the summer.",
    "The lecture covered the basics of thermodynamics.",
]

@torch.no_grad()
def post_norm_states(texts):
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt")
        h = model.model(**enc, use_cache=False).last_hidden_state[0]  # post final norm
        out.append((enc, h))
    return out

# h_bar over generic text, skipping first 4 positions (as in embed_baseline.py)
hs = torch.cat([h[4:] for _, h in post_norm_states(generic)])
h_bar = hs.mean(0)

# phrase prototype: state at token immediately BEFORE the phrase ("before" mode)
vecs = []
for enc, h in post_norm_states(ctx):
    text = tok.decode(enc.input_ids[0])
    offs = tok(text, return_offsets_mapping=True)["offset_mapping"]
    cs = text.index(PHRASE)
    start = next(i for i, (s, e) in enumerate(offs) if s < cs + len(PHRASE) and e > cs)
    vecs.append(h[start - 1])
mu = torch.stack(vecs).mean(0)

def proj_out(v, x):
    return v - (v @ x) / (x @ x) * x
row_proj = F.normalize(proj_out(mu, h_bar), dim=0)          # PRIOR_REPRESENTATION_EMBED_PROJ
row_sub = F.normalize(mu - h_bar, dim=0)                    # PRIOR_REPRESENTATION_EMBED
row_avg = WU[tok.encode(" " + PHRASE)].mean(0)              # AVERAGE_TOKEN_WEIGHTS

print(f"W_U row norms: mean {WU.norm(dim=1).mean():.3f}  median {WU.norm(dim=1).median():.3f}")
print(f"||h_bar|| = {h_bar.norm():.2f}; typical ||h|| = {hs.norm(dim=1).mean():.2f}")
cos_WU = F.cosine_similarity(WU, hs[:200, None, :].mean(0)[None], dim=1)
print(f"cos(W_U rows, mean h): mean {cos_WU.mean():.4f}  max {cos_WU.max():.4f}")
for name, r in [("proj", row_proj), ("sub", row_sub), ("avg", row_avg)]:
    print(f"cos({name} row, individual generic h): mean {F.cosine_similarity(hs, r[None], dim=1).mean():.4f}")

tests = [
    "Fact: The currency used in the country shaped like a boot is",
    "The recipe calls for two cups of",
    "Photosynthesis converts light energy into",
    "The train from Boston arrived late in",
]
print("\nprompt -> [new-row logit | real max logit | rank of new row among V+1]")
for name, r in [("proj", row_proj), ("sub", row_sub), ("avg", row_avg)]:
    print(f"--- {name}")
    for p in tests:
        enc = tok(p, return_tensors="pt")
        with torch.no_grad():
            h = model.model(**enc, use_cache=False).last_hidden_state[0, -1]
        real = WU @ h
        new = r @ h
        rank = int((real > new).sum()) + 1
        top = tok.decode([int(real.argmax())])
        print(f"  {p[:45]:<45} new={new:7.2f}  real_max={real.max():7.2f} ({top!r})  rank={rank}")
