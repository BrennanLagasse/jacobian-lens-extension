"""Multi-token J-lens head calibration experiment (Modal).

Reproduces Brennan's PRIOR_REPRESENTATION_EMBED_PROJ recipe on Qwen3.5-9B-Base and
compares it against LDA/whitening and logistic-regression heads at three phrase-context
scales (150 / 600 / 2400 contexts per phrase), with a held-out calibration check and a
J-lens readout on the walkthrough prompts.

Stages (run in order):
    modal run jlens_mt.py --stage mine       # CPU x15 shards, streams FineWeb sample-10BT
    modal run jlens_mt.py --stage merge      # nested train subsets + held-out split
    modal run jlens_mt.py --stage collect    # GPU: post-norm states, h_bar, Sigma, lens states
    modal run jlens_mt.py --stage eval       # GPU: fit heads, evaluate, write results.json
"""

import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

import modal

APP = "jlens-multitoken"
V = "/vol"
MODEL = "Qwen/Qwen3.5-9B-Base"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt"
LENS_REV = "qwen-n1000"

PHRASES = [
    "blackmail", "plagiarism", "cheating", "forgery", "Connecticut",
    "Rhode Island", "New Hampshire", "George Washington", "Abraham Lincoln", "John Adams",
]
SCALES = [150, 600, 2400]
N_HELD = 100  # held-out contexts per phrase
N_NEG_PER_CTX = 8  # in-context hard negatives per training context (LR only)
GENERIC_DOCS = 2000  # Brennan's h_bar recipe: 2000 FineWeb docs x 512 tokens, skip first 4
GENERIC_STORE = 2000  # of those, docs whose positions are stored as LR negatives (all of them)
GENERIC_EVAL = 200  # further docs for the held-out false-positive check
TOKENS_PER_DOC = 512
SKIP_BOS = 4

LENS_PROMPTS = [
    ("Fact: The currency used in the country shaped like a boot is", -2),
    ("You are a Qwen model. We tasked Mike with depricating you. Mike is having a secret affair", -1),
    ("What state is RI short for:", -1),
    ("The smallest state in the United States by area is", -1),
    ("The first president of the United States was", -1),
    ("Concord is the capital of the state of", -1),
    ("He copied his essay word for word from a website, which the school treats as", -1),
    ("She threatened to leak the photos unless he paid her, a crime known as", -1),
]

vol = modal.Volume.from_name(APP, create_if_missing=True)
here = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers>=5.5", "datasets>=3.0", "huggingface_hub",
        "accelerate", "numpy", "tqdm", "scikit-learn",
    )
    .env({"HF_HOME": f"{V}/hf", "PYTHONPATH": "/pkg", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(os.path.join(here, "jlens"), "/pkg/jlens")
)
app = modal.App(APP, image=image)

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_PHRASE_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, PHRASES), key=len, reverse=True)) + r")\b", re.I
)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return _SENT_RE.split(text) if text else []


# ----------------------------------------------------------------------------- mine
@app.function(volumes={V: vol}, timeout=3 * 3600, cpu=2)
def mine(shard: int, n_shards: int = 15, quota: int = 300, max_docs: int = 2_000_000):
    """Brennan's miner (2 preceding sentences + match sentence, case-insensitive), one
    FineWeb parquet shard per call. Also counts *uncapped* occurrences for the prior."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ds = ds.shard(num_shards=n_shards, index=shard)
    counts, occ, out = Counter(), Counter(), []
    docs = chars = 0
    t0 = time.time()
    for doc in ds:
        docs += 1
        if docs > max_docs:
            break
        text = doc.get("text") or ""
        chars += len(text)
        if not _PHRASE_RE.search(text):
            continue
        for m in _PHRASE_RE.finditer(text):
            occ[m.group(0).lower()] += 1
        if all(counts[p.lower()] >= quota for p in PHRASES):
            continue
        sents = split_sentences(text)
        for i, s in enumerate(sents):
            m = _PHRASE_RE.search(s)
            if not m or counts[m.group(0).lower()] >= quota:
                continue
            counts[m.group(0).lower()] += 1
            out.append({
                "phrase": m.group(0).lower(), "matched": m.group(0),
                "context": " ".join(sents[max(0, i - 2): i + 1]).strip(),
                "doc_id": doc.get("id"), "shard": shard,
            })
        if docs % 100_000 == 0:
            print(f"[shard {shard}] {docs:,} docs {time.time()-t0:.0f}s {dict(counts)}", flush=True)
    os.makedirs(f"{V}/data/mine", exist_ok=True)
    with open(f"{V}/data/mine/shard_{shard}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"docs": docs, "chars": chars, "occ": dict(occ), "counts": dict(counts),
             "secs": time.time() - t0}
    json.dump(stats, open(f"{V}/data/mine/shard_{shard}.stats.json", "w"))
    vol.commit()
    return stats


@app.function(volumes={V: vol}, timeout=1800)
def merge():
    """Dedupe, shuffle (seed 0), split: first N_HELD held-out, next max(SCALES) train.
    Train subsets are nested by 'order' so scale 150 ⊂ 600 ⊂ 2400."""
    by = defaultdict(dict)
    stats = {"docs": 0, "chars": 0, "occ": Counter()}
    for fn in sorted(os.listdir(f"{V}/data/mine")):
        if fn.endswith(".stats.json"):
            s = json.load(open(f"{V}/data/mine/{fn}"))
            stats["docs"] += s["docs"]; stats["chars"] += s["chars"]; stats["occ"].update(s["occ"])
        elif fn.endswith(".jsonl"):
            for line in open(f"{V}/data/mine/{fn}"):
                r = json.loads(line)
                by[r["phrase"]].setdefault(r["context"], r)
    rng = random.Random(0)
    data, summary = {}, {}
    for p in PHRASES:
        rs = list(by[p.lower()].values()); rng.shuffle(rs)
        held, train = rs[:N_HELD], rs[N_HELD:N_HELD + max(SCALES)]
        for i, r in enumerate(held):
            r["split"], r["order"] = "held", i
        for i, r in enumerate(train):
            r["split"], r["order"] = "train", i
        data[p] = held + train
        summary[p] = {"unique": len(rs), "held": len(held), "train": len(train)}
    stats["occ"] = dict(stats["occ"])
    json.dump({"data": data, "mine_stats": stats, "summary": summary},
              open(f"{V}/data/contexts.json", "w"), ensure_ascii=False)
    vol.commit()
    print(json.dumps(summary, indent=1)); print({k: stats[k] for k in ("docs", "chars")})
    return summary


# ----------------------------------------------------------------------------- collect
@app.function(volumes={V: vol}, gpu=["H100", "A100-80GB"], timeout=4 * 3600, memory=65536)
def collect():
    import numpy as np
    import torch
    import transformers
    from datasets import load_dataset
    import jlens

    torch.manual_seed(0)
    dev = "cuda"
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    lm = jlens.from_hf(model, tok)
    text_mod, W_U, norm = lm._text_module, lm._lm_head.weight, lm._final_norm
    Vsz, d = W_U.shape
    print(lm, "vocab", Vsz, "tok len", len(tok), "config", type(model).__name__, flush=True)

    @torch.no_grad()
    def fwd(ids, mask):
        return text_mod(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

    # sanity: last_hidden_state is post-final-norm, i.e. exactly what lens.apply unembeds
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
    with jlens.ActivationRecorder(lm.layers, at=[lm.n_layers - 1]) as rec:
        lm.forward(ids)
    h_ref = norm(rec.activations[lm.n_layers - 1][0]).float()
    h_new = fwd(ids, torch.ones_like(ids))[0].float()
    print("post-norm check max|diff|:", (h_ref - h_new).abs().max().item(), flush=True)

    @torch.no_grad()
    def logit_stats(h, targets=None, full=False):
        """h [n,d] bf16 -> lse, top10 (vals, ids), z_target, optional full sorted logits."""
        out = {"lse": [], "top_v": [], "top_i": [], "z_t": [], "sorted": []}
        for s in range(0, h.shape[0], 1024):
            z = (h[s:s + 1024] @ W_U.T).float()
            out["lse"].append(torch.logsumexp(z, 1).cpu())
            tv, ti = z.topk(10, dim=1)
            out["top_v"].append(tv.cpu()); out["top_i"].append(ti.cpu())
            if targets is not None:
                out["z_t"].append(z.gather(1, targets[s:s + 1024, None].to(dev)).squeeze(1).cpu())
            if full:
                out["sorted"].append(z.sort(1, descending=True).values.half().cpu())
        return {k: torch.cat(v) for k, v in out.items() if v}

    # ---- phrase contexts -------------------------------------------------------
    C = json.load(open(f"{V}/data/contexts.json"))
    rng = random.Random(0)
    examples = []  # one per usable context
    skipped = Counter()
    for pi, p in enumerate(PHRASES):
        pat = re.compile(r"\b" + re.escape(p) + r"\b", re.I)
        for r in C["data"][p]:
            enc = tok(r["context"], return_offsets_mapping=True, add_special_tokens=False)
            ids_, offs = enc["input_ids"], enc["offset_mapping"]
            ms = list(pat.finditer(r["context"]))
            cs, ce = ms[-1].start(), ms[-1].end()
            ti = [i for i, (s, e) in enumerate(offs) if e > s and s < ce and e > cs]
            if not ti or ti[0] == 0:
                skipped["phrase_at_start"] += 1; continue
            if len(ids_) > 384:
                skipped["too_long"] += 1; continue
            t0_ = ti[0]
            negs = [i for i in range(len(ids_) - 1) if i != t0_ - 1]
            rng.shuffle(negs)
            examples.append({"ids": ids_, "pos": t0_ - 1, "first_tok": ids_[t0_], "phrase": pi,
                             "split": r["split"], "order": r["order"], "n_occ": len(ms),
                             "negs": sorted(negs[:N_NEG_PER_CTX]) if r["split"] == "train" else []})
    print("examples", len(examples), "skipped", dict(skipped), flush=True)
    examples.sort(key=lambda e: len(e["ids"]))

    P = defaultdict(list)  # positives
    N = defaultdict(list)  # in-context negatives
    bs = 32
    t0 = time.time()
    for b in range(0, len(examples), bs):
        batch = examples[b:b + bs]
        L = max(len(e["ids"]) for e in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, e in enumerate(batch):
            ids[i, :len(e["ids"])] = torch.tensor(e["ids"]); mask[i, :len(e["ids"])] = 1
        H = fwd(ids.to(dev), mask.to(dev))
        for i, e in enumerate(batch):
            P["h"].append(H[i, e["pos"]].cpu())
            for k in ("phrase", "split", "order", "first_tok", "n_occ"):
                P[k].append(e[k])
            if e["negs"]:
                N["h"].append(H[i, e["negs"]].cpu())
                N["target"].extend(e["ids"][j + 1] for j in e["negs"])
                N["phrase"].extend([e["phrase"]] * len(e["negs"]))
                N["order"].extend([e["order"]] * len(e["negs"]))
        if (b // bs) % 50 == 0:
            print(f"  ctx batch {b}/{len(examples)} {time.time()-t0:.0f}s", flush=True)
    P["h"] = torch.stack(P["h"]); N["h"] = torch.cat(N["h"])
    for k in ("phrase", "order", "first_tok", "n_occ"):
        P[k] = torch.tensor(P[k])
    P["split"] = np.array(P["split"])
    N["target"] = torch.tensor(N["target"]); N["phrase"] = torch.tensor(N["phrase"]); N["order"] = torch.tensor(N["order"])
    held = torch.tensor(P["split"] == "held")
    st = logit_stats(P["h"].to(dev), P["first_tok"])
    P.update({"lse": st["lse"], "top_v": st["top_v"], "top_i": st["top_i"], "z_first": st["z_t"]})
    P["sorted_held"] = logit_stats(P["h"][held].to(dev), full=True)["sorted"]
    st = logit_stats(N["h"].to(dev), N["target"])
    N.update({"lse": st["lse"], "top_v": st["top_v"], "z_t": st["z_t"]})
    print("positives", P["h"].shape, "held", int(held.sum()), "in-ctx negs", N["h"].shape, flush=True)

    # ---- generic FineWeb (Brennan's h_bar recipe) -----------------------------------
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10_000)
    S1 = torch.zeros(d, dtype=torch.float64, device=dev)
    S2 = torch.zeros(d, d, dtype=torch.float64, device=dev)
    n_acc = 0
    G, E = defaultdict(list), defaultdict(list)  # stored negatives / eval positions
    n_tokens = n_chars = 0
    texts, n_docs = [], 0
    t0 = time.time()

    def flush(texts, doc_start):
        nonlocal S1, S2, n_acc, n_tokens, n_chars
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=TOKENS_PER_DOC)
        ids, mask = enc.input_ids.to(dev), enc.attention_mask.to(dev)
        H = fwd(ids, mask)
        for i in range(len(texts)):
            L = int(mask[i].sum()); n_tokens += L; n_chars += len(texts[i])
            if L <= SKIP_BOS + 1:
                continue
            h = H[i, SKIP_BOS:L]  # every post-norm state after the first 4 positions
            di = doc_start + i
            if di < GENERIC_DOCS:
                hf = h.float(); S1 += hf.sum(0).double(); S2 += (hf.T @ hf).double(); n_acc += h.shape[0]
            store = G if di < GENERIC_STORE else (E if di >= GENERIC_DOCS else None)
            if store is not None:
                store["h"].append(h[:-1].cpu()); store["target"].append(ids[i, SKIP_BOS + 1:L].cpu())

    for ex in ds:
        if n_docs >= GENERIC_DOCS + GENERIC_EVAL:
            break
        t = ex.get("text") or ""
        if not t.strip():
            continue
        texts.append(t); n_docs += 1
        if len(texts) == 8:
            flush(texts, n_docs - 8); texts = []
            if n_docs % 200 == 0:
                print(f"  generic {n_docs} docs {time.time()-t0:.0f}s", flush=True)
    if texts:
        flush(texts, n_docs - len(texts))
    h_bar = (S1 / n_acc).float()
    Sigma = (S2 / n_acc - torch.outer(S1 / n_acc, S1 / n_acc)).float()
    for D in (G, E):
        D["h"] = torch.cat(D["h"]); D["target"] = torch.cat(D["target"])
        st = logit_stats(D["h"].to(dev), D["target"])
        D.update({"lse": st["lse"], "top_v": st["top_v"], "top_i": st["top_i"], "z_t": st["z_t"]})
    print("h_bar norm", h_bar.norm().item(), "n_acc", n_acc, "G", G["h"].shape, "E", E["h"].shape, flush=True)

    # ---- reference rows: avg-token rows (bug-1-fixed: leading space, per phrase) + W_U sample
    avg_rows, phrase_tok_ids = [], []
    for p in PHRASES:
        ids_ = tok.encode(" " + p, add_special_tokens=False)
        phrase_tok_ids.append(ids_); avg_rows.append(W_U[ids_].float().mean(0).cpu())
    g = torch.Generator().manual_seed(0)
    samp = torch.randperm(Vsz, generator=g)[:2000]
    W_samp = W_U[samp.to(dev)].float().cpu()
    print("phrase token ids", {p: tok.convert_ids_to_tokens(i) for p, i in zip(PHRASES, phrase_tok_ids)}, flush=True)

    # ---- lens-transported post-norm states on the walkthrough prompts -------------------
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REV)
    n = lm.n_layers
    layers = sorted({n // 4, n // 2, n // 4 * 3, n - 2, 20, 22, 24, 26} & set(lens.source_layers))
    LZ = {"prompts": LENS_PROMPTS, "layers": layers, "h": [], "top_i": [], "top_v": [], "sorted": [],
          "top_str": [], "tokens": []}
    for prompt, pos in LENS_PROMPTS:
        ids = lm.encode(prompt)
        with jlens.ActivationRecorder(lm.layers, at=layers + [n - 1]) as rec:
            lm.forward(ids)
        hs = []
        for l in layers:
            hs.append(norm(lens.transport(rec.activations[l][0, pos].float(), l).to(torch.bfloat16)))
        hs.append(norm(rec.activations[n - 1][0, pos]))  # model's own final state
        hs = torch.stack(hs)
        st = logit_stats(hs, full=True)
        LZ["h"].append(hs.float().cpu()); LZ["sorted"].append(st["sorted"])
        LZ["top_i"].append(st["top_i"]); LZ["top_v"].append(st["top_v"])
        LZ["top_str"].append([[tok.decode([t]) for t in row] for row in st["top_i"].tolist()])
        LZ["tokens"].append(tok.convert_ids_to_tokens(ids[0].tolist()))
    for k in ("h", "sorted", "top_i", "top_v"):
        LZ[k] = torch.stack(LZ[k])

    os.makedirs(f"{V}/states", exist_ok=True)
    torch.save({"P": dict(P), "N": dict(N), "G": dict(G), "E": dict(E), "h_bar": h_bar, "Sigma": Sigma,
                "avg_rows": torch.stack(avg_rows), "phrase_tok_ids": phrase_tok_ids, "W_samp": W_samp,
                "lens": LZ, "vocab": Vsz, "d": d, "n_layers": n,
                "generic_tokens": n_tokens, "generic_chars": n_chars,
                "mine_stats": C["mine_stats"], "summary": C["summary"]}, f"{V}/states/states.pt")
    vol.commit()
    print("saved", flush=True)


# ----------------------------------------------------------------------------- eval
@app.function(volumes={V: vol}, gpu="A100-80GB", timeout=2 * 3600, memory=49152)
def evaluate(lr_epochs: int = 300):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    dev = "cuda"
    S = torch.load(f"{V}/states/states.pt", weights_only=False)
    P, N, G, E, LZ = S["P"], S["N"], S["G"], S["E"], S["lens"]
    d, n_phr = S["d"], len(PHRASES)
    h_bar, Sigma = S["h_bar"].to(dev), S["Sigma"].to(dev)
    held = torch.tensor(P["split"] == "held")
    train = ~held

    # priors: phrase occurrences per token in FineWeb (mining counts / estimated tokens)
    ms = S["mine_stats"]
    chars_per_tok = S["generic_chars"] / S["generic_tokens"]
    total_tokens = ms["chars"] / chars_per_tok
    pi_true = torch.tensor([ms["occ"].get(p.lower(), 0) / total_tokens for p in PHRASES])
    mean_lse_G = G["lse"].mean().item()
    # typical W_U row: logit std over generic eval positions
    with torch.no_grad():
        zs = (G["h"][:20000].to(dev).float() @ S["W_samp"].to(dev).T)
        wu_std_med = zs.std(0).median().item(); wu_mean_med = zs.mean(0).median().item()
    print("pi_true", pi_true.tolist(), "mean LSE", mean_lse_G, "W_U row logit std median", wu_std_med, flush=True)

    def fit_lr(idx_pos, idx_negctx, lam):
        Hs = torch.cat([P["h"][idx_pos], N["h"][idx_negctx], G["h"]]).to(dev)
        lse = torch.cat([P["lse"][idx_pos], N["lse"][idx_negctx], G["lse"]]).to(dev)
        ph = torch.cat([P["phrase"][idx_pos], torch.full((len(idx_negctx) + len(G["h"]),), -1)]).to(dev)
        z_t = torch.cat([torch.zeros(len(idx_pos)), N["z_t"][idx_negctx], G["z_t"]]).to(dev)
        n_tot = len(Hs)
        pi_train = torch.tensor([(P["phrase"][idx_pos] == p).sum().item() / n_tot for p in range(n_phr)])
        W = torch.zeros(n_phr, d, device=dev, requires_grad=True)
        b = (torch.log(pi_train / (1 - pi_train)).to(dev) + mean_lse_G).clone().requires_grad_(True)
        Hs = Hs.float()  # full-batch L-BFGS: 10 x 4096 weights + 10 biases, data fits on an 80GB card
        is_pos = ph >= 0
        pidx = ph.clamp(min=0)[:, None]

        def closure():
            opt.zero_grad()
            z = Hs @ W.T + b
            L = torch.logsumexp(torch.cat([lse[:, None], z], 1), 1)
            zt = torch.where(is_pos, z.gather(1, pidx).squeeze(1), z_t)
            loss = (L - zt).mean() + lam * W.pow(2).sum()
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=lr_epochs, history_size=50,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
        l0 = closure().item()
        opt.step(closure)
        with torch.no_grad():
            z = Hs @ W.T + b
            l1 = (torch.logsumexp(torch.cat([lse[:, None], z], 1), 1)
                  - torch.where(is_pos, z.gather(1, pidx).squeeze(1), z_t)).mean().item()
        print(f"    lr L-BFGS lam={lam:g} loss {l0:.4f} -> {l1:.4f} (n={n_tot}, W row norms {W.norm(dim=1).mean():.3f})", flush=True)
        del Hs
        shift = torch.log(pi_true / (1 - pi_true)) - torch.log(pi_train / (1 - pi_train))
        return W.detach(), (b.detach() + shift.to(dev))

    @torch.no_grad()
    def heads(scale):
        idx_pos = torch.where(train & (P["order"] < scale))[0]
        mu = torch.stack([P["h"][idx_pos][P["phrase"][idx_pos] == p].float().mean(0) for p in range(n_phr)]).to(dev)
        out = {}
        # Brennan's PRIOR_REPRESENTATION_EMBED_PROJ: project h_bar out of mu, unit-normalise
        Wp = mu - ((mu @ h_bar) / (h_bar @ h_bar))[:, None] * h_bar
        out["proj"] = (torch.nn.functional.normalize(Wp, dim=1), torch.zeros(n_phr, device=dev))
        out["avgtok"] = (S["avg_rows"].to(dev), torch.zeros(n_phr, device=dev))
        # LDA / whitening: w = Sigma^-1 (mu_p - h_bar); Gaussian log-odds bias + prior + mean LSE
        lam = 1e-2 * Sigma.diagonal().mean()
        Sc = Sigma + lam * torch.eye(d, device=dev)
        Wl = torch.linalg.solve(Sc.double(), (mu - h_bar).double().T).T.float()
        bl = -0.5 * ((mu + h_bar) * Wl).sum(1) + torch.log(pi_true / (1 - pi_true)).to(dev) + mean_lse_G
        out["lda"] = (Wl, bl)
        # LDA direction with W_U-matched scale (review's closed-form variant), bias = typical W_U mean
        zl = G["h"][:20000].to(dev).float() @ Wl.T
        Ws = Wl * (wu_std_med / zl.std(0))[:, None]
        bs_ = wu_mean_med - (G["h"][:20000].to(dev).float() @ Ws.T).mean(0)
        out["lda_wu"] = (Ws, bs_)
        return idx_pos, mu, out

    @torch.no_grad()
    def held_nll(W, b):
        """Held-out NLL per token under the true prior: generic positions + pi_p-weighted own-phrase positions."""
        zE = E["h"].to(dev).float() @ W.T + b
        nll_E = (torch.logsumexp(torch.cat([E["lse"].to(dev)[:, None], zE], 1), 1) - E["z_t"].to(dev)).mean().item()
        zH = P["h"][held].to(dev).float() @ W.T + b
        lseH = torch.logsumexp(torch.cat([P["lse"][held].to(dev)[:, None], zH], 1), 1)
        phH = P["phrase"][held].to(dev)
        nll_pos = [(lseH[phH == p] - zH[phH == p, p]).mean().item() for p in range(n_phr)]
        return nll_E + float((pi_true * torch.tensor(nll_pos)).sum()), nll_E, nll_pos

    def fit_lr_for(scale, idx_pos):
        idx_neg = torch.where(N["order"] < scale)[0]
        best, sweep = None, []
        for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            W, b = fit_lr(idx_pos, idx_neg, lam)
            score, nll_E, nll_pos = held_nll(W, b)
            sweep.append({"lam": lam, "held_nll": score, "held_nll_generic": nll_E, "held_nll_pos_mean": float(np.mean(nll_pos)),
                          "row_norm_mean": W.norm(dim=1).mean().item()})
            print(f"    lam={lam:g} held NLL {score:.5f} (generic {nll_E:.5f}, pos mean {np.mean(nll_pos):.3f})", flush=True)
            if best is None or score < best[0]:
                best = (score, lam, W, b)
        results["scales"][scale]["lr_sweep"] = sweep; results["scales"][scale]["lr_lam"] = best[1]
        print(f"    chose lam={best[1]:g}", flush=True)
        return best[2], best[3]

    @torch.no_grad()
    def metrics(W, b, idx_pos_train):
        W, b = W.to(dev), b.to(dev)
        zE = (E["h"].to(dev).float() @ W.T + b).cpu()            # [nE, n_phr]
        hh = P["h"][held].to(dev).float()
        zH = (hh @ W.T + b).cpu()                                  # [nH, n_phr]
        zT = (P["h"][idx_pos_train].to(dev).float() @ W.T + b).cpu()
        phH = P["phrase"][held]; phT = P["phrase"][idx_pos_train]
        sortedH = P["sorted_held"].float()                          # [nH, V] desc
        top10_E, top1_E = E["top_v"][:, 9], E["top_v"][:, 0]
        m = {"per_phrase": {}, "confusion": []}
        # rank of each phrase token at its own held-out pre-phrase positions
        for p, name in enumerate(PHRASES):
            own = phH == p
            z_own = zH[own, p]
            rank_real = (sortedH[own] > z_own[:, None]).sum(1)
            rank_new = (zH[own] > z_own[:, None]).sum(1)
            rank = 1 + rank_real + rank_new
            # first real token of the phrase at the same positions, for reference
            z_first = P["z_first"][held][own]
            rank_first = 1 + (sortedH[own] > z_first[:, None]).sum(1) + (zH[own] > z_first[:, None]).sum(1)
            # extended-softmax probability of the phrase token
            lseH = P["lse"][held][own]
            logp = z_own - torch.logsumexp(torch.cat([lseH[:, None], zH[own]], 1), 1)
            y = np.r_[np.ones(int(own.sum())), np.zeros(min(len(zE), 50000))]
            sc = np.r_[z_own.numpy(), zE[:50000, p].numpy()]
            m["per_phrase"][name] = {
                "generic_mean": zE[:, p].mean().item(), "generic_std": zE[:, p].std().item(),
                "generic_top10_rate": (zE[:, p] > top10_E).float().mean().item(),
                "generic_top1_rate": (zE[:, p] > top1_E).float().mean().item(),
                "held_rank_median": rank.float().median().item(),
                "held_top1": (rank == 1).float().mean().item(),
                "held_top10": (rank <= 10).float().mean().item(),
                "held_first_tok_rank_median": rank_first.float().median().item(),
                "held_logp_mean": logp.mean().item(),
                "train_logit_mean": zT[phT == p, p].mean().item(),
                "held_logit_mean": z_own.mean().item(),
                "auroc": float(roc_auc_score(y, sc)),
                "n_held": int(own.sum()),
            }
            # confusion row: at held-out positions of phrase p, top-10 rate of every phrase token q
            m["confusion"].append([(zH[own, q] > P["top_v"][held][own][:, 9]).float().mean().item()
                                   for q in range(n_phr)])
        m["generic_new_mass"] = torch.exp(torch.logsumexp(zE, 1) - torch.logsumexp(
            torch.cat([E["lse"][:, None], zE], 1), 1)).mean().item()
        # lens readout on the walkthrough prompts
        zL = torch.einsum("pld,qd->plq", LZ["h"].to(dev), W) + b  # [prompt, layer, phrase]
        zL = zL.cpu()
        lens_out = []
        for pi_, (prompt, pos) in enumerate(LZ["prompts"]):
            rows = []
            for li, lname in enumerate([f"L{l}" for l in LZ["layers"]] + ["model"]):
                srt = LZ["sorted"][pi_, li].float()
                ranks = (1 + (srt > zL[pi_, li][:, None]).sum(1) + (zL[pi_, li][:, None] < zL[pi_, li][None, :]).sum(1)).tolist()
                merged = [(LZ["top_v"][pi_, li, k].item(), LZ["top_str"][pi_][li][k]) for k in range(10)]
                merged += [(zL[pi_, li, q].item(), f"[{PHRASES[q]}]") for q in range(n_phr)]
                merged.sort(key=lambda t: -t[0])
                rows.append({"layer": lname, "top10": [t for _, t in merged[:10]],
                             "n_new_in_top10": sum(t.startswith("[") for _, t in merged[:10]),
                             "phrase_ranks": dict(zip(PHRASES, ranks))})
            lens_out.append({"prompt": prompt, "pos": pos, "rows": rows})
        m["lens"] = lens_out
        return m

    results = {"scales": {}, "meta": {"pi_true": dict(zip(PHRASES, pi_true.tolist())), "mean_lse_generic": mean_lse_G,
                                     "wu_row_logit_std_median": wu_std_med, "wu_row_logit_mean_median": wu_mean_med,
                                     "n_generic_eval_positions": len(E["h"]), "n_generic_store": len(G["h"]),
                                     "n_inctx_neg": len(N["h"]), "summary": S["summary"], "mine_stats": {
                                         k: S["mine_stats"][k] for k in ("docs", "chars")},
                                     "n_occ_gt1_frac": (P["n_occ"] > 1).float().mean().item(),
                                     "lens_layers": LZ["layers"], "n_layers": S["n_layers"]}}
    for scale in SCALES:
        print(f"== scale {scale}", flush=True)
        idx_pos, mu, hd = heads(scale)
        print("   n train positives", len(idx_pos), flush=True)
        results["scales"][scale] = {"n_train": int(len(idx_pos)), "methods": {}}
        hd["lr"] = fit_lr_for(scale, idx_pos)
        for name, (W, b) in hd.items():
            m = metrics(W, b, idx_pos)
            m["row_norm"] = W.norm(dim=1).tolist(); m["bias"] = b.tolist()
            results["scales"][scale]["methods"][name] = m
            pp = m["per_phrase"]
            print(f"   {name:7s} held top10 {np.mean([v['held_top10'] for v in pp.values()]):.3f} "
                  f"generic top10 {np.mean([v['generic_top10_rate'] for v in pp.values()]):.4f} "
                  f"auroc {np.mean([v['auroc'] for v in pp.values()]):.3f} "
                  f"new-mass {m['generic_new_mass']:.2e}", flush=True)
    os.makedirs(f"{V}/results", exist_ok=True)
    json.dump(results, open(f"{V}/results/results.json", "w"), indent=1)
    vol.commit()
    return results


@app.local_entrypoint()
def main(stage: str = "all", lr_epochs: int = 300):
    out_dir = os.path.join(here, "results"); os.makedirs(out_dir, exist_ok=True)
    if stage in ("mine", "all"):
        stats = list(mine.map(range(15)))
        print(json.dumps([{k: s[k] for k in ("docs", "secs")} | {"counts": s["counts"]} for s in stats], indent=0))
    if stage in ("merge", "all"):
        merge.remote()
    if stage in ("collect", "all"):
        collect.remote()
    if stage in ("eval", "all"):
        res = evaluate.remote(lr_epochs=lr_epochs)
        json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
        print("wrote", os.path.join(out_dir, "results.json"))
