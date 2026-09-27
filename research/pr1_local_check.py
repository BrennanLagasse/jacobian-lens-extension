import sys, random, torch
sys.path.insert(0, "../brennan/multi_token")
from transformers import AutoModelForCausalLM, AutoTokenizer
from extend_model import generate_extended_tok_and_model, EmbedMethod
M = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
V0 = len(tok)
model, etok = generate_extended_tok_and_model(model, tok, EmbedMethod.FITTED_ROWS, "results/phrase_rows.pt")
# tokenizer-wrapper tests (same assertions as test_extended_model.py)
assert etok.encode("blackmail") == tok.encode("blackmail")
rid = random.randint(0, V0 - 1); assert etok.decode([rid]) == tok.decode([rid])
assert etok.decode([V0]) == next(iter(torch.load("results/phrase_rows.pt")["phrase_rows"]))
assert model.get_output_embeddings().weight.shape[0] == len(etok) == V0 + 2
print("wrapper tests ok; new ids", V0, V0 + 1, "->", etok.decode([V0]), "|", etok.decode([V0 + 1]))
for p in ["The train from Boston arrived late in", "Fact: The currency used in the country shaped like a boot is",
          "The recipe calls for two cups of", "She was born in", "The president of the"]:
    with torch.no_grad():
        z = model(**tok(p, return_tensors="pt")).logits[0, -1]
    top = z.topk(10).indices.tolist()
    ranks = {etok.decode([i]): int((z > z[i]).sum()) + 1 for i in (V0, V0 + 1)}
    print(f"{p[:45]:<45} top10={[etok.decode([i]) for i in top]}  ranks={ranks}")
