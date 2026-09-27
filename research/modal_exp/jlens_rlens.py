"""R-lens (RelP-modified Jacobian lens) on Qwen3.5-9B-Base, with the attribution experiment (Modal).

R-lens = the jlens estimator with LRP rules installed in the backward pass (camilablank et al.,
"R-lens: making J-lens more faithful on early layers"; provenance of their qwen3.5-9b/r-lens:
ln_rule, identity_rule, half_rule beta=0.5, q/k norms and gated norms untouched, target_layer 30,
skip_first 4, 25 prompts from NeelNanda/pile-10k at 128 tokens). Forward values are unchanged; only
gradients differ, so J and R are a matched pair.

Stages:
    modal run jlens_rlens.py --stage fit         # fit J-lens and R-lens (matched recipe) on the Base model
    modal run jlens_rlens.py --stage attribution # pass@10 + direction-ablation on two-hop questions
    modal run jlens_rlens.py --stage phrases     # R-lens vs J-lens vs logit-lens profiles on the phrase benchmark
"""

import json
import math
import os
import time
import types
from collections import defaultdict

import modal

APP = "jlens-rlens"
V = "/vol"
MODEL = "Qwen/Qwen3.5-9B-Base"
TARGET_LAYER = 30
N_FIT_PROMPTS = 25
FIT_SEQ = 128
SKIP_FIRST = 4
HALF_BETA = 0.5

# Two-hop questions for a base model. Readout position is the penultimate token (the word before "is").
# intermediate: aliases of the latent entity (first alias is the one whose direction is ablated);
# answer: aliases accepted in the sampled continuation.
MULTIHOP = [
    ("Fact: The currency used in the country shaped like a boot is", ["Italy", "Italian"], ["euro", "lira"]),
    ("Fact: The capital of the country where the Eiffel Tower stands is", ["France", "French"], ["Paris"]),
    ("Fact: The official language of the country whose capital is Madrid is", ["Spain"], ["Spanish", "Castilian"]),
    ("Fact: The currency of the country famous for the Great Pyramids is", ["Egypt", "Egyptian"], ["pound"]),
    ("Fact: The capital of the country that hosted the 2016 Summer Olympics is", ["Brazil", "Rio"], ["Bras", "Brasilia", "Brasília"]),
    ("Fact: The largest city in the state whose capital is Sacramento is", ["California"], ["Los Angeles"]),
    ("Fact: The capital of the country where Mount Fuji is located is", ["Japan", "Japanese"], ["Tokyo"]),
    ("Fact: The currency of the country where the Kremlin stands is", ["Russia", "Russian", "Moscow"], ["ruble", "rouble"]),
    ("Fact: The continent containing the country whose capital is Nairobi is", ["Kenya"], ["Africa"]),
    ("Fact: The capital of the country that has the Taj Mahal is", ["India", "Indian"], ["Delhi"]),
    ("Fact: The national language of the country whose capital is Berlin is", ["Germany"], ["German"]),
    ("Fact: The capital of the country where the Colosseum stands is", ["Italy", "Italian"], ["Rome"]),
    ("Fact: The currency of the country where Big Ben is located is", ["Britain", "England", "UK", "United"], ["pound", "sterling"]),
    ("Fact: The largest city of the country whose capital is Canberra is", ["Australia"], ["Sydney"]),
    ("Fact: The capital of the country that produces the most maple syrup is", ["Canada", "Canadian"], ["Ottawa"]),
    ("Fact: The capital of the country whose flag is a red circle on a white field is", ["Japan", "Japanese"], ["Tokyo"]),
    ("Fact: The currency of the country where the Acropolis stands is", ["Greece", "Greek", "Athens"], ["euro", "drachma"]),
    ("Fact: The capital of the country where Machu Picchu is located is", ["Peru", "Peruvian"], ["Lima"]),
    ("Fact: The capital of the country where the Sydney Opera House stands is", ["Australia", "Australian"], ["Canberra"]),
    ("Fact: The official language of the country whose capital is Lisbon is", ["Portugal"], ["Portuguese"]),
    ("Fact: The capital of the country where the Sphinx of Giza is located is", ["Egypt", "Egyptian"], ["Cairo"]),
    ("Fact: The currency of the country whose capital is Bern is", ["Switzerland", "Swiss"], ["franc"]),
    ("Fact: The largest city in the state that borders both New York and Ohio is", ["Pennsylvania"], ["Philadelphia"]),
    ("Fact: The capital of the state where Disney World is located is", ["Florida"], ["Tallahassee"]),
    ("Fact: The capital of the state where the Golden Gate Bridge is located is", ["California"], ["Sacramento"]),
    ("Fact: The capital of the state whose largest city is Chicago is", ["Illinois"], ["Springfield"]),
    ("Fact: The capital of the state where Mount Rushmore is located is", ["South", "Dakota"], ["Pierre"]),
    ("Fact: The capital of the country where the Brandenburg Gate stands is", ["Germany", "German"], ["Berlin"]),
    ("Fact: The nationality of the painter of the Mona Lisa is", ["Leonardo", "Vinci"], ["Italian"]),
    ("Fact: The home country of the author of Hamlet is", ["Shakespeare"], ["England", "Britain", "United Kingdom", "English"]),
    ("Fact: The currency of the country where the Louvre is located is", ["France", "French", "Paris"], ["euro", "franc"]),
    ("Fact: The capital of the most populous country in South America is", ["Brazil"], ["Bras", "Brasilia", "Brasília"]),
    ("Fact: The capital of the country where the Great Barrier Reef is located is", ["Australia", "Australian"], ["Canberra"]),
    ("Fact: The official language of the country where the Kremlin stands is", ["Russia", "Russian", "Moscow"], ["Russian"]),
    ("Fact: The capital of the country where Angkor Wat is located is", ["Cambodia", "Cambodian"], ["Phnom"]),
    ("Fact: The currency of the country where the Forbidden City is located is", ["China", "Chinese", "Beijing"], ["yuan", "renminbi", "RMB"]),
    ("Fact: The capital of the country where Petra is located is", ["Jordan"], ["Amman"]),
    ("Fact: The capital of the country where the Blue Mosque of Istanbul is located is", ["Turkey", "Turkish"], ["Ankara"]),
    ("Fact: The capital of the country whose national dish is paella is", ["Spain", "Spanish"], ["Madrid"]),
    ("Fact: The official language of the country whose capital is Beijing is", ["China"], ["Chinese", "Mandarin"]),
    ("Fact: The currency of the country whose capital is Ottawa is", ["Canada", "Canadian"], ["dollar"]),
    ("Fact: The largest city in the state whose capital is Austin is", ["Texas"], ["Houston"]),
    ("Fact: The capital of the state whose largest city is Seattle is", ["Washington"], ["Olympia"]),
    ("Fact: The capital of the state where the Grand Canyon is located is", ["Arizona"], ["Phoenix"]),
    ("Fact: The capital of the state where Las Vegas is located is", ["Nevada"], ["Carson"]),
    ("Fact: The capital of the country where Chernobyl is located is", ["Ukraine", "Ukrainian"], ["Kyiv", "Kiev"]),
    ("Fact: The currency of the country where the Taj Mahal is located is", ["India", "Indian"], ["rupee"]),
    ("Fact: The capital of the country where the Parthenon stands is", ["Greece", "Greek"], ["Athens"]),
    ("Fact: The capital of the country where the Alhambra is located is", ["Spain", "Spanish"], ["Madrid"]),
    ("Fact: The capital of the country where Stonehenge is located is", ["England", "Britain", "United"], ["London"]),
]

vol = modal.Volume.from_name("jlens-multitoken", create_if_missing=True)
here = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=5.5", "datasets>=3.0", "huggingface_hub", "accelerate", "numpy", "tqdm", "scikit-learn")
    .env({"HF_HOME": f"{V}/hf", "PYTHONPATH": "/pkg", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(os.path.join(here, "jlens"), "/pkg/jlens")
)
app = modal.App(APP, image=image)


# ----------------------------------------------------------------------------- shared helpers
def load_model(dev="cuda"):
    import torch
    import transformers
    import jlens

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    lm = jlens.from_hf(model, tok)
    return tok, model, lm


def install_relp(lm, beta=HALF_BETA):
    """Install the RelP backward rules on the residual RMSNorms and gated MLPs (instance-level overrides).
    Forward values are unchanged up to bf16 rounding of silu = g*sigmoid(g). Returns an undo function."""
    import torch

    patched = []

    def graft(value, relp):
        """Forward value = the standard forward (bit-identical); gradient = the RelP expression's gradient."""
        return value.detach() + (relp - relp.detach())

    def rms_forward(self, x):  # LN-rule: normalisation scale treated as a constant in the backward pass
        xf = x.float()
        with torch.no_grad():
            std = type(self).forward(self, x)
        scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps).detach()
        return graft(std, (xf * scale * (1.0 + self.weight.float())).type_as(x))

    def mlp_forward(self, x):  # identity-rule on SiLU, half-rule on the SwiGLU product
        g, u = self.gate_proj(x), self.up_proj(x)
        with torch.no_grad():
            std = self.act_fn(g) * u
        a = g * torch.sigmoid(g).detach()
        prod = beta * (a * u.detach()) + (1.0 - beta) * (a.detach() * u)
        return self.down_proj(graft(std, prod))

    assert lm._hf_model.config.get_text_config().hidden_act == "silu"
    for layer in lm.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            m = getattr(layer, name)
            assert type(m).__name__.endswith("RMSNorm"), type(m).__name__
            m.forward = types.MethodType(rms_forward, m); patched.append(m)
        mlp = layer.mlp
        assert hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"), type(mlp).__name__
        mlp.forward = types.MethodType(mlp_forward, mlp); patched.append(mlp)
    fn = lm._final_norm
    fn.forward = types.MethodType(rms_forward, fn); patched.append(fn)

    def undo():
        for m in patched:
            del m.forward
    return undo


def load_fit_prompts(tok, n=N_FIT_PROMPTS):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    out = []
    for ex in ds:
        t = ex["text"]
        if len(tok(t, add_special_tokens=False).input_ids) >= 32:
            out.append(t)
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------------------- fit
@app.function(volumes={V: vol}, gpu="H100", timeout=6 * 3600, memory=65536)
def fit(dim_batch: int = 16):
    import torch
    import jlens

    jlens.configure_logging()
    tok, model, lm = load_model()
    prompts = load_fit_prompts(tok)
    os.makedirs(f"{V}/lenses", exist_ok=True)
    # sanity: RelP forward is (numerically) the standard forward
    ids = lm.encode(prompts[0], max_length=64)
    with torch.no_grad():
        h0 = lm.forward(ids).last_hidden_state.float()
    undo = install_relp(lm)
    with torch.no_grad():
        h1 = lm.forward(ids).last_hidden_state.float()
    undo()
    print("relp forward max|diff| / |h|:", ((h1 - h0).abs().max() / h0.abs().max()).item(), flush=True)

    for name, relp in (("jlens", False), ("rlens", True)):
        path = f"{V}/lenses/base_{name}_t{TARGET_LAYER}.pt"
        if os.path.exists(path):
            print("exists", path); continue
        undo = install_relp(lm) if relp else (lambda: None)
        t0 = time.time()
        lens = jlens.fit(lm, prompts, target_layer=TARGET_LAYER, dim_batch=dim_batch, max_seq_len=FIT_SEQ,
                         skip_first=SKIP_FIRST, checkpoint_path=f"{V}/lenses/ckpt_{name}.pt", checkpoint_every=5)
        undo()
        lens.save(path)
        vol.commit()
        print(f"{name}: fitted {lens.n_prompts} prompts in {time.time()-t0:.0f}s -> {path}", flush=True)


# ----------------------------------------------------------------------------- attribution
@app.function(volumes={V: vol}, gpu="H100", timeout=4 * 3600, memory=65536)
def attribution(n_samples: int = 8, temperature: float = 0.7, max_new: int = 12):
    import numpy as np
    import torch
    import jlens

    torch.manual_seed(0)
    dev = "cuda"
    tok, model, lm = load_model()
    n_layers, d = lm.n_layers, lm.d_model
    W_U, norm = lm._lm_head.weight, lm._final_norm
    gamma = (1.0 + norm.weight.float())
    vol.reload()
    lenses = {name: jlens.JacobianLens.load(f"{V}/lenses/base_{name}_t{TARGET_LAYER}.pt") for name in ("jlens", "rlens")}
    Js = {name: {l: L.jacobians[l].to(dev) for l in L.source_layers} for name, L in lenses.items()}

    def transport(name, h, l):
        if name == "logit" or l not in Js[name]:
            return h
        return h @ Js[name][l].T

    @torch.no_grad()
    def readout_topk(h, k=10):
        return (norm(h.to(torch.bfloat16)).float() @ W_U.float().T).topk(k, dim=-1).indices

    # --- pass@10: is the intermediate's first token in the lens top-10 at the penultimate position? ---
    items = []
    for prompt, inter, ans in MULTIHOP:
        ids = lm.encode(prompt)
        inter_ids = [tok.encode(" " + a, add_special_tokens=False)[0] for a in inter]
        items.append({"prompt": prompt, "ids": ids, "inter": inter, "inter_ids": inter_ids, "answer": ans})
    layers = list(range(n_layers))
    pass10 = {name: {pos: np.zeros(n_layers) for pos in (-2, -1)} for name in ("rlens", "jlens", "logit")}
    with torch.no_grad():
        for it in items:
            with jlens.ActivationRecorder(lm.layers, at=layers) as rec:
                lm.forward(it["ids"])
            for pos in (-2, -1):
                for l in layers:
                    h = rec.activations[l][0, pos].float()
                    for name in pass10:
                        top = readout_topk(transport(name, h, l)).tolist()
                        pass10[name][pos][l] += any(t in top for t in it["inter_ids"])
    n_items = len(items)
    pass10 = {name: {str(pos): (v / n_items).tolist() for pos, v in d_.items()} for name, d_ in pass10.items()}
    print("pass@10 at -2 (R/J/logit) by layer:", flush=True)
    for l in layers:
        print(f"  L{l:2d} R={pass10['rlens']['-2'][l]:.2f} J={pass10['jlens']['-2'][l]:.2f} logit={pass10['logit']['-2'][l]:.2f}", flush=True)

    # --- direction ablation ---
    g = torch.Generator(device="cpu").manual_seed(1)
    rand_dirs = torch.nn.functional.normalize(torch.randn(n_layers, d, generator=g), dim=1).to(dev)

    def directions(name, tok_id):
        """Unit residual-stream direction at every layer whose removal kills the lens readout of tok_id."""
        row = gamma * W_U[tok_id].float()
        out = []
        for l in range(n_layers):
            if name == "random":
                out.append(rand_dirs[l]); continue
            dl = row if (name == "logit" or l not in Js[name]) else Js[name][l].T @ row
            out.append(torch.nn.functional.normalize(dl, dim=0))
        return torch.stack(out)  # [n_layers, d]

    state = {"dirs": None, "layers": set(), "pos": None}
    handles = []
    for l, block in enumerate(lm.layers):
        def hook(module, inputs, output, l=l):
            if state["dirs"] is None or l not in state["layers"]:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] == 1:  # decoding step; the prompt position was ablated during prefill
                return output
            dvec = state["dirs"][l].to(hs.dtype)
            for p in state["pos"]:
                hp = hs[:, p]
                hs[:, p] = hp - (hp @ dvec)[:, None] * dvec
            return output
        handles.append(block.register_forward_hook(hook))

    def accuracy(it):
        ids = it["ids"]
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=temperature, top_p=0.95,
                                 max_new_tokens=max_new, num_return_sequences=n_samples, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        hits = [any(a.lower() in t.lower() for a in it["answer"]) for t in texts]
        return float(np.mean(hits)), texts

    half = set(range(n_layers // 2)); allL = set(range(n_layers))
    conditions = [(name, lname, lset, pos) for name in ("rlens", "jlens", "logit", "random")
                  for lname, lset in (("first_half", half), ("all", allL)) for pos in ((-2,), (-2, -1))]
    res = {"items": [], "pass10": pass10, "conditions": [f"{c[0]}|{c[1]}|{c[3]}" for c in conditions]}
    for it in items:
        state["dirs"] = None
        base, base_texts = accuracy(it)
        rec = {"prompt": it["prompt"], "answer": it["answer"], "intermediate": it["inter"][0], "base_acc": base,
               "base_samples": base_texts[:3], "abl": {}}
        for name, lname, lset, pos in conditions:
            state["dirs"] = directions(name, it["inter_ids"][0]); state["layers"] = lset; state["pos"] = pos
            acc, texts = accuracy(it)
            rec["abl"][f"{name}|{lname}|{pos}"] = {"acc": acc, "sample": texts[0]}
        state["dirs"] = None
        res["items"].append(rec)
        print(f"  base {base:.2f} | " + " ".join(f"{k.split('|')[0][:3]}{'H' if 'first' in k else 'A'}{len(eval(k.split('|')[2]))}={v['acc']:.2f}"
                                               for k, v in rec["abl"].items()) + f" | {it['prompt'][6:50]}", flush=True)
    for h in handles:
        h.remove()
    # summary: relative accuracy loss over questions the model gets right at baseline (>= 0.5)
    good = [r for r in res["items"] if r["base_acc"] >= 0.5]
    res["summary"] = {"n_questions": len(res["items"]), "n_baseline_correct": len(good),
                      "mean_base_acc_all": float(np.mean([r["base_acc"] for r in res["items"]]))}
    for c in res["conditions"]:
        accs = np.array([r["abl"][c]["acc"] for r in good]); bases = np.array([r["base_acc"] for r in good])
        res["summary"][c] = {"mean_acc_after": float(accs.mean()), "mean_acc_before": float(bases.mean()),
                             "relative_loss": float(1 - accs.mean() / bases.mean()),
                             "per_q_relative_loss_mean": float(np.mean(1 - accs / bases))}
        print(f"  {c:28s} acc {bases.mean():.3f} -> {accs.mean():.3f}  relative loss {1 - accs.mean()/bases.mean():.3f}", flush=True)
    os.makedirs(f"{V}/results", exist_ok=True)
    json.dump(res, open(f"{V}/results/rlens_attribution.json", "w"), indent=1)
    vol.commit()
    return res


# ----------------------------------------------------------------------------- phrase benchmark under R-lens
PHRASES = ["blackmail", "plagiarism", "cheating", "forgery", "Connecticut", "Rhode Island", "New Hampshire",
           "George Washington", "Abraham Lincoln", "John Adams", "New York", "New Jersey", "John Quincy Adams", "Massachusetts", "Vermont"]
LENS_LAYERS = [4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 30]


@app.function(volumes={V: vol}, gpu="H100", timeout=3 * 3600, memory=131072)
def phrases(lam: float = 1e-6, scale: int = 2400):
    """Held-out lens-layer profiles of the phrase benchmark under the matched J-lens / R-lens pair (target 30)
    and the logit lens, for the real first token, Brennan's head and the bias-free logistic phrase head."""
    import re
    import numpy as np
    import torch
    import jlens

    torch.manual_seed(0)
    dev = "cuda"
    tok, model, lm = load_model()
    W_U, norm, text_mod = lm._lm_head.weight, lm._final_norm, lm._text_module
    n_phr, d = len(PHRASES), lm.d_model
    S = torch.load(f"{V}/states/bench.pt", weights_only=False)
    P, N, G = S["P"], S["N"], S["G"]
    held = torch.tensor(P["split"] == "held"); train = ~held
    h_bar = S["h_bar"].to(dev)
    ms = S["mine_stats"]
    total_tokens = ms["chars"] / (S["generic_chars"] / S["generic_tokens"])
    pi_true = torch.tensor([ms["occ"].get(p.lower(), 0) / total_tokens for p in PHRASES])

    # --- refit the bias-free logistic head (as in jlens_bench.py) and Brennan's head at the given scale ---
    idx_pos = torch.where(train & (P["order"] < scale))[0]; idx_neg = torch.where(N["order"] < scale)[0]
    Hs = torch.cat([P["h"][idx_pos], N["h"][idx_neg], G["h"]]).to(dev).float()
    lse = torch.cat([P["lse"][idx_pos], N["lse"][idx_neg], G["lse"]]).to(dev)
    ph = torch.cat([P["phrase"][idx_pos], torch.full((len(idx_neg) + len(G["h"]),), -1)]).to(dev)
    z_t = torch.cat([torch.zeros(len(idx_pos)), N["z_t"][idx_neg], G["z_t"]]).to(dev)
    n_tot = len(Hs)
    pi_train = torch.tensor([(ph == i).sum().item() / n_tot for i in range(n_phr)])
    is_pos = ph >= 0; pidx = ph.clamp(min=0)[:, None]
    wts = torch.where(is_pos, (pi_true / pi_train).to(dev)[pidx.squeeze(1)], torch.ones(n_tot, device=dev)).double(); wsum = wts.sum()
    W = torch.zeros(n_phr, d, device=dev, requires_grad=True)

    def closure():
        opt.zero_grad()
        z = Hs @ W.T
        L = torch.logsumexp(torch.cat([lse[:, None], z], 1), 1)
        zt = torch.where(is_pos, z.gather(1, pidx).squeeze(1), z_t)
        loss = ((L - zt).double() * wts).sum() / wsum + lam * W.pow(2).sum()
        loss.backward(); return loss
    opt = torch.optim.LBFGS([W], lr=1, max_iter=300, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
    opt.step(closure)
    W_lr = W.detach()
    mu = torch.stack([P["h"][idx_pos][P["phrase"][idx_pos] == p].float().mean(0) for p in range(n_phr)]).to(dev)
    Wp = mu - ((mu @ h_bar) / (h_bar @ h_bar))[:, None] * h_bar
    W_proj = torch.nn.functional.normalize(Wp, dim=1)
    del Hs
    print("heads refit; lr row norms", W_lr.norm(dim=1).mean().item(), flush=True)

    # --- held-out contexts, re-tokenised exactly as in the benchmark collect stage ---
    C = json.load(open(f"{V}/data/contexts_bench.json"))
    ex = []
    for pi, p in enumerate(PHRASES):
        pat = re.compile(r"\b" + re.escape(p) + r"\b", re.I)
        for r in C["data"][p]:
            if r["split"] != "held":
                continue
            enc = tok(r["context"], return_offsets_mapping=True, add_special_tokens=False)
            ids_, offs = enc["input_ids"], enc["offset_mapping"]
            m = list(pat.finditer(r["context"]))[-1]
            ti = [i for i, (s_, e_) in enumerate(offs) if e_ > s_ and s_ < m.end() and e_ > m.start()]
            if not ti or ti[0] == 0 or len(ids_) > 384:
                continue
            ex.append({"ids": ids_, "pos": ti[0] - 1, "first": ids_[ti[0]], "phrase": pi})
    ex.sort(key=lambda e: len(e["ids"]))
    print("held-out contexts", len(ex), flush=True)

    vol.reload()
    lenses = {name: jlens.JacobianLens.load(f"{V}/lenses/base_{name}_t{TARGET_LAYER}.pt") for name in ("jlens", "rlens")}
    Js = {name: {l: L.jacobians[l].to(dev, torch.bfloat16) for l in L.source_layers} for name, L in lenses.items()}
    names = ("rlens", "jlens", "logit")
    nL = len(LENS_LAYERS)
    acc = {name: {"first_top10": [], "phr_lr_top10": [], "phr_proj_top10": [], "first_rank_bucket": []} for name in names}
    phr_idx = []
    grid = torch.tensor([0, 1, 2, 4, 9, 19, 49, 99, 199, 499, 999], device=dev)  # ranks 1,2,3,5,10,20,50,100,200,500,1000
    with torch.no_grad():
        for b in range(0, len(ex), 16):
            batch = ex[b:b + 16]
            L = max(len(e["ids"]) for e in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long); mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, e in enumerate(batch):
                ids[i, :len(e["ids"])] = torch.tensor(e["ids"]); mask[i, :len(e["ids"])] = 1
            with jlens.ActivationRecorder(lm.layers, at=LENS_LAYERS) as rec:
                text_mod(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
            pos = torch.tensor([e["pos"] for e in batch], device=dev); ar = torch.arange(len(batch), device=dev)
            first = torch.tensor([e["first"] for e in batch], device=dev); phr = torch.tensor([e["phrase"] for e in batch], device=dev)
            phr_idx.extend(e["phrase"] for e in batch)
            for name in names:
                hs = []
                for l in LENS_LAYERS:
                    r_ = rec.activations[l][ar, pos]
                    hs.append(norm(r_ if (name == "logit" or l not in Js[name]) else r_ @ Js[name][l].T))
                hs = torch.stack(hs, 1).float()                                  # [B, nL, d]
                z = hs @ W_U.float().T                                           # [B, nL, V]
                srt = z.sort(-1, descending=True).values
                thr = srt[..., grid]                                             # [B, nL, 11]
                z_first = z.gather(-1, first[:, None, None].expand(-1, nL, 1)).squeeze(-1)
                z_lr = torch.einsum("bld,pd->blp", hs, W_lr); z_proj = torch.einsum("bld,pd->blp", hs, W_proj)
                zp_lr = z_lr[ar, :, phr]; zp_proj = z_proj[ar, :, phr]
                top10 = thr[..., 4]                                              # 10th largest real logit
                acc[name]["first_top10"].append((z_first >= top10).cpu())
                acc[name]["phr_lr_top10"].append(((zp_lr > top10) & ((z_lr > zp_lr[..., None]).sum(-1) + (thr > zp_lr[..., None]).sum(-1) < 10)).cpu())
                acc[name]["phr_proj_top10"].append(((zp_proj > top10) & ((z_proj > zp_proj[..., None]).sum(-1) + (thr > zp_proj[..., None]).sum(-1) < 10)).cpu())
                acc[name]["first_rank_bucket"].append((thr > z_first[..., None]).sum(-1).cpu())
            if (b // 16) % 40 == 0:
                print(f"  batch {b}/{len(ex)}", flush=True)
    phr_idx = torch.tensor(phr_idx)
    out = {"layers": LENS_LAYERS, "n_held": len(ex), "lam": lam, "scale": scale, "profiles": {}}
    for name in names:
        A = {k: torch.cat(v).float() for k, v in acc[name].items()}
        ft, pl, pp = A["first_top10"], A["phr_lr_top10"], A["phr_proj_top10"]
        out["profiles"][name] = {
            "first_top10_by_layer": ft.mean(0).tolist(),
            "phrase_lr_top10_by_layer": pl.mean(0).tolist(),
            "phrase_proj_top10_by_layer": pp.mean(0).tolist(),
            "phrase_lr_only_by_layer": (pl * (1 - ft)).mean(0).tolist(),
            "first_top1_by_layer": (A["first_rank_bucket"] == 0).float().mean(0).tolist(),
            "per_phrase_first_top10": {PHRASES[p]: ft[phr_idx == p].mean(0).tolist() for p in range(n_phr)},
            "per_phrase_lr_top10": {PHRASES[p]: pl[phr_idx == p].mean(0).tolist() for p in range(n_phr)},
        }
        print(name, "first-token top-10 by layer", [f"{x:.2f}" for x in out["profiles"][name]["first_top10_by_layer"]], flush=True)
        print(name, "phrase (LR no bias) top-10  ", [f"{x:.2f}" for x in out["profiles"][name]["phrase_lr_top10_by_layer"]], flush=True)
    os.makedirs(f"{V}/results", exist_ok=True)
    json.dump(out, open(f"{V}/results/rlens_phrases.json", "w"), indent=1)
    vol.commit()
    return out


@app.local_entrypoint()
def main(stage: str = "fit"):
    out_dir = os.path.join(here, "results"); os.makedirs(out_dir, exist_ok=True)
    if stage == "fit":
        fit.remote()
    elif stage == "attribution":
        res = attribution.remote()
        json.dump(res, open(os.path.join(out_dir, "rlens_attribution.json"), "w"), indent=1)
        print("wrote", os.path.join(out_dir, "rlens_attribution.json"))
    elif stage == "phrases":
        res = phrases.remote()
        json.dump(res, open(os.path.join(out_dir, "rlens_phrases.json"), "w"), indent=1)
