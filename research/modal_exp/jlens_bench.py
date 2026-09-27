"""Phrase-head benchmark for the multi-token J-lens (Modal).

Frozen evaluation set: Brennan's 10 phrases + 5 confusables (New York, New Jersey, John Quincy Adams,
Massachusetts, Vermont), 200 held-out contexts per phrase split into single-occurrence / copy buckets,
500 held-out generic FineWeb documents. Heads (proj, avg W_U rows, LDA, LDA W_U-scaled, logistic
regression, random-row floor, real-W_U-row ceiling) are scored on direction (own vs generic, own vs
first-token sibling, own vs category sibling), calibration (prior-weighted ECE, mass vs prior, NLL),
rank behaviour (vs the phrase's real first token), copy sensitivity, J-lens vs logit-lens layer
profiles over all held-out contexts, and stability over three disjoint 150-context refits.

Stages:
    modal run jlens_bench.py --stage mine      # only the 5 new phrases (old shards are reused)
    modal run jlens_bench.py --stage merge
    modal run jlens_bench.py --stage collect
    modal run jlens_bench.py --stage eval
"""

import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

import modal

APP = "jlens-bench"
V = "/vol"
MODEL = "Qwen/Qwen3.5-9B-Base"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt"
LENS_REV = "qwen-n1000"

ORIG = ["blackmail", "plagiarism", "cheating", "forgery", "Connecticut", "Rhode Island", "New Hampshire",
        "George Washington", "Abraham Lincoln", "John Adams"]
NEW = ["New York", "New Jersey", "John Quincy Adams", "Massachusetts", "Vermont"]
PHRASES = ORIG + NEW
CATEGORY = {"crime": ["blackmail", "plagiarism", "cheating", "forgery"],
            "state": ["Connecticut", "Rhode Island", "New Hampshire", "New York", "New Jersey", "Massachusetts", "Vermont"],
            "president": ["George Washington", "Abraham Lincoln", "John Adams", "John Quincy Adams"]}
FIRST_SIB = {"New Hampshire": ["New York", "New Jersey"], "New York": ["New Hampshire", "New Jersey"],
             "New Jersey": ["New Hampshire", "New York"], "John Adams": ["John Quincy Adams"], "John Quincy Adams": ["John Adams"]}
SCALES = [150, 600, 2400]
N_HELD = 200
N_NEG_PER_CTX = 8
GENERIC_DOCS = 2000
GENERIC_EVAL = 500
TOKENS_PER_DOC = 512
SKIP_BOS = 4
LENS_LAYERS = [4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 30]
RANK_GRID = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

PROMPTS = [
    ("Fact: The currency used in the country shaped like a boot is", -2),
    ("You are a Qwen model. We tasked Mike with depricating you. Mike is having a secret affair", -1),
    ("What state is RI short for:", -1),
    ("The smallest state in the United States by area is", -1),
    ("The first president of the United States was", -1),
    ("Concord is the capital of the state of", -1),
    ("He copied his essay word for word from a website, which the school treats as", -1),
    ("She threatened to leak the photos unless he paid her, a crime known as", -1),
    ("The Statue of Liberty stands in the harbor of", -1),
    ("Trenton is the capital of", -1),
    ("Providence is the capital of", -1),
    ("Boston is the capital of", -1),
    ("Montpelier is the capital of", -1),
    ("The second president of the United States was", -1),
    ("The sixth president of the United States, son of the second, was", -1),
    ("The president who issued the Emancipation Proclamation was", -1),
    ("He signed another person's name on the check, which is the crime of", -1),
    ("The student was caught with the answers hidden in his sleeve, which is", -1),
    ("The four states of New England with the smallest populations are Maine,", -1),
    ("Yale University is located in New Haven,", -1),
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

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


def phrase_re(phrases):
    return re.compile(r"\b(" + "|".join(sorted(map(re.escape, phrases), key=len, reverse=True)) + r")\b", re.I)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return _SENT_RE.split(text) if text else []


# ----------------------------------------------------------------------------- mine (new phrases only)
@app.function(volumes={V: vol}, timeout=3 * 3600, cpu=2)
def mine(shard: int, n_shards: int = 15, quota: int = 300, out_dir: str = "mine2"):
    from datasets import load_dataset

    RE = phrase_re(NEW)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ds = ds.shard(num_shards=n_shards, index=shard)
    counts, occ, out = Counter(), Counter(), []
    docs = chars = 0
    t0 = time.time()
    for doc in ds:
        docs += 1
        text = doc.get("text") or ""
        chars += len(text)
        if not RE.search(text):
            continue
        for m in RE.finditer(text):
            occ[m.group(0).lower()] += 1
        if all(counts[p.lower()] >= quota for p in NEW):
            continue
        sents = split_sentences(text)
        for i, s in enumerate(sents):
            m = RE.search(s)
            if not m or counts[m.group(0).lower()] >= quota:
                continue
            counts[m.group(0).lower()] += 1
            out.append({"phrase": m.group(0).lower(), "matched": m.group(0),
                        "context": " ".join(sents[max(0, i - 2): i + 1]).strip(), "doc_id": doc.get("id"), "shard": shard})
        if docs % 200_000 == 0:
            print(f"[shard {shard}] {docs:,} docs {time.time()-t0:.0f}s {dict(counts)}", flush=True)
    os.makedirs(f"{V}/data/{out_dir}", exist_ok=True)
    with open(f"{V}/data/{out_dir}/shard_{shard}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"docs": docs, "chars": chars, "occ": dict(occ), "counts": dict(counts), "secs": time.time() - t0}
    json.dump(stats, open(f"{V}/data/{out_dir}/shard_{shard}.stats.json", "w"))
    vol.commit()
    return stats


@app.function(volumes={V: vol}, timeout=1800)
def merge():
    by = defaultdict(dict)
    stats = {"docs": 0, "chars": 0, "occ": Counter()}
    for d in ("mine", "mine2"):
        for fn in sorted(os.listdir(f"{V}/data/{d}")):
            if fn.endswith(".stats.json"):
                s = json.load(open(f"{V}/data/{d}/{fn}"))
                if d == "mine":
                    stats["docs"] += s["docs"]; stats["chars"] += s["chars"]
                stats["occ"].update(s["occ"])
            elif fn.endswith(".jsonl"):
                for line in open(f"{V}/data/{d}/{fn}"):
                    r = json.loads(line)
                    by[r["phrase"]].setdefault(r["context"], r)
    rng = random.Random(1)
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
    json.dump({"data": data, "mine_stats": stats, "summary": summary}, open(f"{V}/data/contexts_bench.json", "w"), ensure_ascii=False)
    vol.commit()
    print(json.dumps(summary, indent=1))
    return summary


# ----------------------------------------------------------------------------- collect
@app.function(volumes={V: vol}, gpu=["H100", "A100-80GB"], timeout=4 * 3600, memory=131072)
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
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REV)
    J = {l: lens.jacobians[l].to(dev, torch.bfloat16) for l in LENS_LAYERS}
    print(lm, "vocab", Vsz, "lens layers", lens.source_layers[:3], "...", flush=True)
    grid = torch.tensor([r - 1 for r in RANK_GRID], device=dev)

    @torch.no_grad()
    def fwd(ids, mask):
        return text_mod(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

    @torch.no_grad()
    def logit_stats(h, targets=None, full=False):
        """h [n,d] -> lse, top10 vals/ids, z_target, rank-grid thresholds, optional full sorted logits."""
        out = defaultdict(list)
        for s in range(0, h.shape[0], 1024):
            z = (h[s:s + 1024].to(torch.bfloat16) @ W_U.T).float()
            out["lse"].append(torch.logsumexp(z, 1).cpu())
            srt = z.sort(1, descending=True).values
            out["top_v"].append(srt[:, :10].cpu()); out["top_i"].append(z.topk(10, 1).indices.cpu())
            out["thr"].append(srt[:, grid].cpu())
            if targets is not None:
                out["z_t"].append(z.gather(1, targets[s:s + 1024, None].to(dev)).squeeze(1).cpu())
            if full:
                out["sorted"].append(srt.half().cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    # ---- contexts -------------------------------------------------------------------------
    C = json.load(open(f"{V}/data/contexts_bench.json"))
    rng = random.Random(0)
    examples, skipped = [], Counter()
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
            negs = [i for i in range(len(ids_) - 1) if i != t0_ - 1]; rng.shuffle(negs)
            examples.append({"ids": ids_, "pos": t0_ - 1, "first_tok": ids_[t0_], "phrase": pi, "split": r["split"],
                             "order": r["order"], "n_occ": len(ms),
                             "negs": sorted(negs[:N_NEG_PER_CTX]) if r["split"] == "train" else []})
    print("examples", len(examples), "skipped", dict(skipped), flush=True)

    P, N, HL = defaultdict(list), defaultdict(list), defaultdict(list)
    for split in ("train", "held"):
        ex = sorted([e for e in examples if e["split"] == split], key=lambda e: len(e["ids"]))
        bs = 32 if split == "train" else 16
        t0 = time.time()
        for b in range(0, len(ex), bs):
            batch = ex[b:b + bs]
            L = max(len(e["ids"]) for e in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, e in enumerate(batch):
                ids[i, :len(e["ids"])] = torch.tensor(e["ids"]); mask[i, :len(e["ids"])] = 1
            if split == "held":
                with jlens.ActivationRecorder(lm.layers, at=LENS_LAYERS) as rec:
                    H = fwd(ids.to(dev), mask.to(dev))
                pos = torch.tensor([e["pos"] for e in batch], device=dev)
                ar = torch.arange(len(batch), device=dev)
                hj, hl = [], []
                for l in LENS_LAYERS:
                    r_ = rec.activations[l][ar, pos]                      # [B, d] residual at layer l
                    hj.append(norm(r_ @ J[l].T)); hl.append(norm(r_))    # J-lens / logit-lens post-norm states
                HL["hj"].append(torch.stack(hj, 1).cpu()); HL["hl"].append(torch.stack(hl, 1).cpu())  # [B, nL, d]
            else:
                H = fwd(ids.to(dev), mask.to(dev))
            for i, e in enumerate(batch):
                P["h"].append(H[i, e["pos"]].cpu())
                for k in ("phrase", "split", "order", "first_tok", "n_occ"):
                    P[k].append(e[k])
                if e["negs"]:
                    N["h"].append(H[i, e["negs"]].cpu())
                    N["target"].extend(e["ids"][j + 1] for j in e["negs"])
                    N["phrase"].extend([e["phrase"]] * len(e["negs"])); N["order"].extend([e["order"]] * len(e["negs"]))
            if (b // bs) % 100 == 0:
                print(f"  {split} batch {b}/{len(ex)} {time.time()-t0:.0f}s", flush=True)
    P["h"] = torch.stack(P["h"]); N["h"] = torch.cat(N["h"])
    for k in ("phrase", "order", "first_tok", "n_occ"):
        P[k] = torch.tensor(P[k])
    P["split"] = np.array(P["split"])
    for k in ("target", "phrase", "order"):
        N[k] = torch.tensor(N[k])
    held = torch.tensor(P["split"] == "held")
    st = logit_stats(P["h"].to(dev), P["first_tok"])
    P.update({"lse": st["lse"], "top_v": st["top_v"], "top_i": st["top_i"], "z_first": st["z_t"], "thr": st["thr"]})
    P["sorted_held"] = logit_stats(P["h"][held].to(dev), full=True)["sorted"]
    st = logit_stats(N["h"].to(dev), N["target"])
    N.update({"lse": st["lse"], "top_v": st["top_v"], "z_t": st["z_t"]})
    # lens states of held-out contexts: [nH, nL, d] each, plus per-(ctx, layer) real-logit stats
    HL["hj"] = torch.cat(HL["hj"]); HL["hl"] = torch.cat(HL["hl"])
    nH, nL = HL["hj"].shape[:2]
    ft = P["first_tok"][held].repeat_interleave(nL)
    for key in ("hj", "hl"):
        st = logit_stats(HL[key].reshape(-1, d).to(dev), ft)
        for k in ("lse", "top_v", "z_t", "thr"):
            HL[f"{key}_{k}"] = st[k].reshape(nH, nL, -1).squeeze(-1)
    print("positives", P["h"].shape, "held", int(held.sum()), "negs", N["h"].shape, "lens", HL["hj"].shape, flush=True)

    # ---- generic ----------------------------------------------------------------------------
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    S1 = torch.zeros(d, dtype=torch.float64, device=dev); S2 = torch.zeros(d, d, dtype=torch.float64, device=dev)
    n_acc = 0
    G, E = defaultdict(list), defaultdict(list)
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
            h = H[i, SKIP_BOS:L]
            di = doc_start + i
            if di < GENERIC_DOCS:
                hf = h.float(); S1 += hf.sum(0).double(); S2 += (hf.T @ hf).double(); n_acc += h.shape[0]
            store = G if di < GENERIC_DOCS else E
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
            if n_docs % 500 == 0:
                print(f"  generic {n_docs} docs {time.time()-t0:.0f}s", flush=True)
    if texts:
        flush(texts, n_docs - len(texts))
    h_bar = (S1 / n_acc).float()
    Sigma = (S2 / n_acc - torch.outer(S1 / n_acc, S1 / n_acc)).float()
    for D in (G, E):
        D["h"] = torch.cat(D["h"]); D["target"] = torch.cat(D["target"])
        st = logit_stats(D["h"].to(dev), D["target"])
        D.update({"lse": st["lse"], "top_v": st["top_v"], "z_t": st["z_t"]})
    print("h_bar", h_bar.norm().item(), "G", G["h"].shape, "E", E["h"].shape, flush=True)

    # ---- reference rows ----------------------------------------------------------------------
    avg_rows, phrase_tok_ids, real_rows = [], [], {}
    for pi, p in enumerate(PHRASES):
        ids_ = tok.encode(" " + p, add_special_tokens=False)
        phrase_tok_ids.append(ids_); avg_rows.append(W_U[ids_].float().mean(0).cpu())
        if len(ids_) == 1:
            real_rows[pi] = W_U[ids_[0]].float().cpu()
    g = torch.Generator().manual_seed(0)
    W_samp = W_U[torch.randperm(Vsz, generator=g)[:2000].to(dev)].float().cpu()
    print("phrase tokens", {p: tok.convert_ids_to_tokens(i) for p, i in zip(PHRASES, phrase_tok_ids)}, flush=True)

    # ---- hand prompts -----------------------------------------------------------------------
    n = lm.n_layers
    PL = [l for l in [8, 16, 20, 22, 24, 26, 30] if l in lens.source_layers]
    LZ = {"prompts": PROMPTS, "layers": PL, "h": [], "top_v": [], "sorted": [], "top_str": []}
    for prompt, pos in PROMPTS:
        ids = lm.encode(prompt)
        with jlens.ActivationRecorder(lm.layers, at=PL + [n - 1]) as rec:
            lm.forward(ids)
        hs = [norm(lens.transport(rec.activations[l][0, pos].float(), l).to(torch.bfloat16)) for l in PL]
        hs.append(norm(rec.activations[n - 1][0, pos]))
        hs = torch.stack(hs)
        st = logit_stats(hs, full=True)
        LZ["h"].append(hs.float().cpu()); LZ["sorted"].append(st["sorted"]); LZ["top_v"].append(st["top_v"])
        LZ["top_str"].append([[tok.decode([t]) for t in row] for row in st["top_i"].tolist()])
    for k in ("h", "sorted", "top_v"):
        LZ[k] = torch.stack(LZ[k])

    os.makedirs(f"{V}/states", exist_ok=True)
    torch.save({"P": dict(P), "N": dict(N), "G": dict(G), "E": dict(E), "HL": dict(HL), "h_bar": h_bar, "Sigma": Sigma,
                "avg_rows": torch.stack(avg_rows), "real_rows": real_rows, "phrase_tok_ids": phrase_tok_ids, "W_samp": W_samp,
                "lens": LZ, "vocab": Vsz, "d": d, "generic_tokens": n_tokens, "generic_chars": n_chars,
                "mine_stats": C["mine_stats"], "summary": C["summary"]}, f"{V}/states/bench.pt")
    vol.commit()
    print("saved", flush=True)


# ----------------------------------------------------------------------------- eval
@app.function(volumes={V: vol}, gpu="A100-80GB", timeout=3 * 3600, memory=131072)
def evaluate():
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    dev = "cuda"
    S = torch.load(f"{V}/states/bench.pt", weights_only=False)
    P, N, G, E, HL, LZ = S["P"], S["N"], S["G"], S["E"], S["HL"], S["lens"]
    d, n_phr = S["d"], len(PHRASES)
    h_bar, Sigma = S["h_bar"].to(dev), S["Sigma"].to(dev)
    held = torch.tensor(P["split"] == "held"); train = ~held
    phH = P["phrase"][held]; n_occH = P["n_occ"][held]
    hH = P["h"][held].to(dev).float()
    hE = E["h"].to(dev).float()
    lseE, ztE, topE = E["lse"].to(dev), E["z_t"].to(dev), E["top_v"].to(dev)
    sortedH = P["sorted_held"].float()  # CPU [nH, V]
    rank_grid = torch.tensor(RANK_GRID)

    ms = S["mine_stats"]
    chars_per_tok = S["generic_chars"] / S["generic_tokens"]
    total_tokens = ms["chars"] / chars_per_tok
    pi_true = torch.tensor([ms["occ"].get(p.lower(), 0) / total_tokens for p in PHRASES])
    mean_lse_G = G["lse"].mean().item()
    with torch.no_grad():
        zs = G["h"][:20000].to(dev).float() @ S["W_samp"].to(dev).T
        wu_std_med, wu_mean_med = zs.std(0).median().item(), zs.mean(0).median().item()
    print("pi_true", [f"{x:.1e}" for x in pi_true.tolist()], flush=True)

    # ---- heads ----------------------------------------------------------------------------------
    def fit_lr(idx_pos, idx_neg, lam, phr_subset=None, bias=True):
        sub = list(range(n_phr)) if phr_subset is None else list(phr_subset)
        remap = {p: i for i, p in enumerate(sub)}
        keep = torch.tensor([int(p) in remap for p in P["phrase"][idx_pos].tolist()])
        idx_pos = idx_pos[keep]
        Hs = torch.cat([P["h"][idx_pos], N["h"][idx_neg], G["h"]]).to(dev).float()
        lse = torch.cat([P["lse"][idx_pos], N["lse"][idx_neg], G["lse"]]).to(dev)
        ph = torch.cat([torch.tensor([remap[int(p)] for p in P["phrase"][idx_pos]]), torch.full((len(idx_neg) + len(G["h"]),), -1)]).to(dev)
        z_t = torch.cat([torch.zeros(len(idx_pos)), N["z_t"][idx_neg], G["z_t"]]).to(dev)
        n_tot, k = len(Hs), len(sub)
        pi_train = torch.tensor([(ph == i).sum().item() / n_tot for i in range(k)])
        W = torch.zeros(k, d, device=dev, requires_grad=True)
        is_pos = ph >= 0; pidx = ph.clamp(min=0)[:, None]
        if bias:
            b = (torch.log(pi_train / (1 - pi_train)).to(dev) + mean_lse_G).clone().requires_grad_(True)
            wts = torch.ones(n_tot, device=dev, dtype=torch.float64)
            params = [W, b]
        else:
            # no bias (like a real W_U row): reweight positives to the FineWeb prior instead of shifting a bias afterwards
            b = torch.zeros(k, device=dev)
            wr = (pi_true[sub] / pi_train).to(dev)
            wts = torch.where(is_pos, wr[pidx.squeeze(1)], torch.ones(n_tot, device=dev)).double()
            params = [W]
        wsum = wts.sum()

        def closure():
            opt.zero_grad()
            z = Hs @ W.T + b
            L = torch.logsumexp(torch.cat([lse[:, None], z], 1), 1)
            zt = torch.where(is_pos, z.gather(1, pidx).squeeze(1), z_t)
            loss = ((L - zt).double() * wts).sum() / wsum + lam * W.pow(2).sum()
            loss.backward(); return loss

        opt = torch.optim.LBFGS(params, lr=1, max_iter=300, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15)
        opt.step(closure)
        shift = (torch.log(pi_true[sub] / (1 - pi_true[sub])) - torch.log(pi_train / (1 - pi_train))) if bias else torch.zeros(k)
        Wf = torch.zeros(n_phr, d, device=dev); bf = torch.full((n_phr,), -1e4, device=dev)  # absent phrases: -inf-ish
        Wf[sub] = W.detach(); bf[sub] = b.detach() + shift.to(dev)
        del Hs
        return Wf, bf

    @torch.no_grad()
    def closed_form(idx_pos):
        mu = torch.stack([P["h"][idx_pos][P["phrase"][idx_pos] == p].float().mean(0) if (P["phrase"][idx_pos] == p).any()
                          else torch.zeros(d) for p in range(n_phr)]).to(dev)
        out = {}
        Wp = mu - ((mu @ h_bar) / (h_bar @ h_bar))[:, None] * h_bar
        out["proj"] = (torch.nn.functional.normalize(Wp, dim=1), torch.zeros(n_phr, device=dev))
        Sc = Sigma + 1e-2 * Sigma.diagonal().mean() * torch.eye(d, device=dev)
        Wl = torch.linalg.solve(Sc.double(), (mu - h_bar).double().T).T.float()
        bl = -0.5 * ((mu + h_bar) * Wl).sum(1) + torch.log(pi_true / (1 - pi_true)).to(dev) + mean_lse_G
        out["lda"] = (Wl, bl)
        zl = G["h"][:20000].to(dev).float() @ Wl.T
        Ws = Wl * (wu_std_med / zl.std(0))[:, None]
        out["lda_wu"] = (Ws, wu_mean_med - (G["h"][:20000].to(dev).float() @ Ws.T).mean(0))
        return out

    def reference_heads():
        out = {"avgtok": (S["avg_rows"].to(dev), torch.zeros(n_phr, device=dev))}
        g = torch.Generator().manual_seed(123)
        out["random"] = (torch.nn.functional.normalize(torch.randn(n_phr, d, generator=g), dim=1).to(dev) * wu_std_med / 1.0,
                         torch.zeros(n_phr, device=dev))
        Wr = S["avg_rows"].clone()
        for pi_, row in S["real_rows"].items():
            Wr[pi_] = row
        out["real_wu"] = (Wr.to(dev), torch.zeros(n_phr, device=dev))
        return out

    # ---- metrics ----------------------------------------------------------------------------------
    def rank_from_grid(thr, z):
        """thr [.., len(RANK_GRID)] sorted-desc logit at ranks RANK_GRID; z [..] -> rank bucket lower bound."""
        # number of grid ranks whose threshold logit is > z  ->  rank >= RANK_GRID[k]
        k = (thr > z[..., None]).sum(-1)
        return torch.where(k == 0, torch.ones_like(k), rank_grid.to(k.device)[(k - 1).clamp(min=0)] + 1)

    @torch.no_grad()
    def metrics(W, b, quick=False):
        W, b = W.to(dev), b.to(dev)
        zE = hE @ W.T + b                       # [nE, n_phr]
        zH = (hH @ W.T + b).cpu()               # [nH, n_phr]
        top10E, top1E = topE[:, 9], topE[:, 0]
        m = {"per_phrase": {}, "confusion": []}
        for p, name in enumerate(PHRASES):
            own = phH == p
            z_own = zH[own, p]
            rank_real = (sortedH[own] > z_own[:, None]).sum(1)
            rank_new = (zH[own] > z_own[:, None]).sum(1)
            rank = 1 + rank_real + rank_new
            z_first = P["z_first"][held][own]
            rank_first = 1 + (sortedH[own] > z_first[:, None]).sum(1) + (zH[own] > z_first[:, None]).sum(1)
            lseH = P["lse"][held][own]
            logp = z_own - torch.logsumexp(torch.cat([lseH[:, None], zH[own]], 1), 1)
            single = (n_occH[own] == 1)
            r = {}
            for bucket, sel in (("all", torch.ones_like(single)), ("single", single), ("copy", ~single)):
                if sel.sum() < 5:
                    continue
                zE_p = zE[:60000, p].cpu()
                y = np.r_[np.ones(int(sel.sum())), np.zeros(len(zE_p))]
                sc = np.r_[z_own[sel].numpy(), zE_p.numpy()]
                r[bucket] = {"n": int(sel.sum()), "auroc_generic": float(roc_auc_score(y, sc)),
                             "rank_median": rank[sel].float().median().item(), "top1": (rank[sel] == 1).float().mean().item(),
                             "top10": (rank[sel] <= 10).float().mean().item(),
                             "first_rank_median": rank_first[sel].float().median().item(),
                             "log_rank_ratio_median": torch.log(rank[sel].float() / rank_first[sel].float()).median().item(),
                             "logp_mean": logp[sel].mean().item()}
                sibs = [PHRASES.index(s) for s in FIRST_SIB.get(name, [])]
                if sibs:
                    sib_pos = torch.isin(phH, torch.tensor(sibs))
                    y = np.r_[np.ones(int(sel.sum())), np.zeros(int(sib_pos.sum()))]
                    sc = np.r_[z_own[sel].numpy(), zH[sib_pos, p].numpy()]
                    r[bucket]["auroc_first_sib"] = float(roc_auc_score(y, sc))
                    r[bucket]["top10_at_sib"] = (zH[sib_pos, p] > P["top_v"][held][sib_pos][:, 9]).float().mean().item()
                cat = [PHRASES.index(s) for c in CATEGORY.values() if name in c for s in c if s != name]
                cat_pos = torch.isin(phH, torch.tensor(cat))
                y = np.r_[np.ones(int(sel.sum())), np.zeros(int(cat_pos.sum()))]
                r[bucket]["auroc_cat_sib"] = float(roc_auc_score(y, np.r_[z_own[sel].numpy(), zH[cat_pos, p].numpy()]))
            # prior-weighted reliability: predicted p(phrase) on own positions (weight pi) and generic (weight 1-pi)
            pE = torch.exp(zE[:, p] - torch.logsumexp(torch.cat([lseE[:, None], zE], 1), 1)).cpu()
            pH = torch.exp(logp)
            pi = pi_true[p].item()
            edges = torch.tensor([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0001])
            ece = 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                wH = ((pH >= lo) & (pH < hi)).float().sum() / len(pH) * pi
                wE = ((pE >= lo) & (pE < hi)).float().sum() / len(pE) * (1 - pi)
                if wH + wE == 0:
                    continue
                conf = (pH[(pH >= lo) & (pH < hi)].sum() / len(pH) * pi + pE[(pE >= lo) & (pE < hi)].sum() / len(pE) * (1 - pi)) / (wH + wE)
                acc = wH / (wH + wE)
                ece += float((wH + wE) * abs(conf - acc))
            r["ece"] = ece
            r["generic_mean"] = zE[:, p].mean().item(); r["generic_std"] = zE[:, p].std().item()
            r["generic_top10"] = (zE[:, p] > top10E).float().mean().item(); r["generic_top1"] = (zE[:, p] > top1E).float().mean().item()
            r["mass_ratio"] = (pE.mean() / pi).item()
            m["per_phrase"][name] = r
            m["confusion"].append([(zH[own, q] > P["top_v"][held][own][:, 9]).float().mean().item() for q in range(n_phr)])
        m["generic_new_mass"] = torch.exp(torch.logsumexp(zE, 1) - torch.logsumexp(torch.cat([lseE[:, None], zE], 1), 1)).mean().item()
        m["generic_nll"] = (torch.logsumexp(torch.cat([lseE[:, None], zE], 1), 1) - ztE).mean().item()
        if quick:
            return m
        # lens-layer profiles over all held-out contexts: rank bucket of phrase token per layer, J-lens vs logit lens
        prof = {}
        for key, lab in (("hj", "jlens"), ("hl", "logit_lens")):
            zL = torch.einsum("nld,pd->nlp", HL[key].to(dev).float(), W).cpu() + b.cpu()   # [nH, nL, n_phr]
            thr = HL[f"{key}_thr"]                                                       # [nH, nL, grid]
            zown = zL[torch.arange(len(phH)), :, phH]                                    # [nH, nL]
            rk = rank_from_grid(thr, zown) + (zL > zown[..., None]).sum(-1)              # + new-token competitors
            rk_first = rank_from_grid(thr, HL[f"{key}_z_t"]) + (zL > HL[f"{key}_z_t"][..., None]).sum(-1)
            top10 = (rk <= 10).float(); top10f = (rk_first <= 10).float()
            first_layer = torch.where(top10.any(1), top10.argmax(1), torch.full((len(phH),), len(LENS_LAYERS)))
            first_layer_f = torch.where(top10f.any(1), top10f.argmax(1), torch.full((len(phH),), len(LENS_LAYERS)))
            prof[lab] = {
                "layers": LENS_LAYERS,
                "phrase_top10_by_layer": top10.mean(0).tolist(),
                "first_tok_top10_by_layer": top10f.mean(0).tolist(),
                "phrase_top1_by_layer": (rk == 1).float().mean(0).tolist(),
                "phrase_only_by_layer": (top10 * (1 - top10f)).mean(0).tolist(),
                "both_by_layer": (top10 * top10f).mean(0).tolist(),
                "earliest_top10_layer_median": {PHRASES[p]: (LENS_LAYERS + [99])[int(first_layer[phH == p].median())] for p in range(n_phr)},
                "never_top10_frac": {PHRASES[p]: (first_layer[phH == p] == len(LENS_LAYERS)).float().mean().item() for p in range(n_phr)},
                "earliest_top10_layer_median_first_tok": {PHRASES[p]: (LENS_LAYERS + [99])[int(first_layer_f[phH == p].median())] for p in range(n_phr)},
                "per_phrase_top10_by_layer": {PHRASES[p]: top10[phH == p].mean(0).tolist() for p in range(n_phr)},
            }
        m["lens_profile"] = prof
        # hand prompts
        zP = torch.einsum("pld,qd->plq", LZ["h"].to(dev), W).cpu() + b.cpu()
        outp = []
        for i, (prompt, pos) in enumerate(LZ["prompts"]):
            rows = []
            for li, lname in enumerate([f"L{l}" for l in LZ["layers"]] + ["model"]):
                merged = [(LZ["top_v"][i, li, k].item(), LZ["top_str"][i][li][k]) for k in range(10)]
                merged += [(zP[i, li, q].item(), f"[{PHRASES[q]}]") for q in range(n_phr)]
                merged.sort(key=lambda t: -t[0])
                rows.append({"layer": lname, "top10": [t for _, t in merged[:10]], "n_new": sum(t.startswith("[") for _, t in merged[:10])})
            outp.append({"prompt": prompt, "rows": rows})
        m["prompts"] = outp
        return m

    def summarize(m):
        pp = m["per_phrase"]
        return {"auroc_generic": np.mean([v["all"]["auroc_generic"] for v in pp.values()]),
                "top10": np.mean([v["all"]["top10"] for v in pp.values()]),
                "generic_top10": np.mean([v["generic_top10"] for v in pp.values()]),
                "new_mass": m["generic_new_mass"], "ece": np.mean([v["ece"] for v in pp.values()]),
                "rank_median": np.median([v["all"]["rank_median"] for v in pp.values()])}

    results = {"meta": {"phrases": PHRASES, "pi_true": dict(zip(PHRASES, pi_true.tolist())), "prior_sum": float(pi_true.sum()),
                        "mean_lse_generic": mean_lse_G, "wu_row_logit_std_median": wu_std_med, "wu_row_logit_mean_median": wu_mean_med,
                        "n_generic_eval_positions": len(E["h"]), "n_generic_store": len(G["h"]), "n_held": int(held.sum()),
                        "held_single_frac": (n_occH == 1).float().mean().item(), "summary": S["summary"],
                        "mine_docs": ms["docs"], "single_token_phrases": [PHRASES[i] for i in S["real_rows"]],
                        "lens_layers": LENS_LAYERS, "rank_grid": RANK_GRID},
               "scales": {}, "stability": {}, "orig10": {}}
    ref = reference_heads()
    for scale in SCALES:
        print(f"== scale {scale}", flush=True)
        idx_pos = torch.where(train & (P["order"] < scale))[0]
        idx_neg = torch.where(N["order"] < scale)[0]
        heads = dict(closed_form(idx_pos))
        n_pos_tr, n_neg_tr = len(idx_pos), len(idx_neg) + len(G["h"])
        w_val = (n_pos_tr / n_neg_tr) * (len(E["h"]) / int(held.sum()))  # reweight held sets to the training pos:neg mix

        def val_loss(W, b, mq):
            with torch.no_grad():
                zH_ = hH @ W.T + b
                nll_H = torch.logsumexp(torch.cat([P["lse"][held].to(dev)[:, None], zH_], 1), 1) - zH_[torch.arange(len(phH)), phH.to(dev)]
                return (mq["generic_nll"] * len(E["h"]) + w_val * nll_H.sum().item()) / (len(E["h"]) + w_val * len(phH)), nll_H.mean().item()

        for tag, lams, bias in (("lr", (1e-5, 1e-4, 1e-3), True), ("lr_nobias", (1e-6, 1e-7, 1e-8), False)):
            best = None
            for lam in lams:
                W, b = fit_lr(idx_pos, idx_neg, lam, bias=bias)
                mq = metrics(W, b, quick=True)
                val, own = val_loss(W, b, mq)
                print(f"   {tag} lam={lam:g} generic NLL {mq['generic_nll']:.5f} own NLL {own:.3f} val {val:.5f} auroc {summarize(mq)['auroc_generic']:.4f}", flush=True)
                heads[f"{tag}_{lam:g}"] = (W, b)
                if best is None or val < best[0]:
                    best = (val, lam, W, b)
            heads[tag] = (best[2], best[3])
            results.setdefault("lam_choice", {}).setdefault(str(scale), {})[tag] = best[1]
        if scale == SCALES[0]:
            heads.update(ref)
        results["scales"][scale] = {"n_train": int(len(idx_pos)), "lr_lam": results["lam_choice"][str(scale)]["lr"],
                                    "lr_nobias_lam": results["lam_choice"][str(scale)]["lr_nobias"], "methods": {}}
        for name, (W, b) in heads.items():
            m = metrics(W, b)
            m["row_norm"] = W.norm(dim=1).tolist(); m["bias"] = b.tolist()
            results["scales"][scale]["methods"][name] = m
            s = summarize(m)
            print(f"   {name:8s} " + " ".join(f"{k}={v:.4f}" for k, v in s.items()), flush=True)

    print("== stability (3 disjoint 150-context refits)", flush=True)
    for name in ("proj", "lda", "lr", "lr_nobias"):
        runs = []
        for k in range(3):
            idx_pos = torch.where(train & (P["order"] >= 150 * k) & (P["order"] < 150 * (k + 1)))[0]
            idx_neg = torch.where((N["order"] >= 150 * k) & (N["order"] < 150 * (k + 1)))[0]
            W, b = (closed_form(idx_pos)[name] if name in ("proj", "lda")
                    else fit_lr(idx_pos, idx_neg, results["scales"][150]["lr_lam" if name == "lr" else "lr_nobias_lam"], bias=(name == "lr")))
            runs.append(summarize(metrics(W, b, quick=True)))
        results["stability"][name] = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
        print(f"   {name}: " + " ".join(f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in results["stability"][name].items()), flush=True)

    print("== heads fitted on the original 10 phrases only, scored at sibling positions", flush=True)
    idx_pos = torch.where(train & (P["order"] < 2400))[0]
    idx_neg = torch.where(N["order"] < 2400)[0]
    orig_idx = [PHRASES.index(p) for p in ORIG]
    keep = torch.isin(P["phrase"][idx_pos], torch.tensor(orig_idx))
    heads10 = {k: v for k, v in closed_form(idx_pos[keep]).items() if k in ("proj", "lda")}
    heads10["lr"] = fit_lr(idx_pos, idx_neg, results["scales"][2400]["lr_lam"], phr_subset=orig_idx)
    heads10["lr_nobias"] = fit_lr(idx_pos, idx_neg, results["scales"][2400]["lr_nobias_lam"], phr_subset=orig_idx, bias=False)
    for name, (W, b) in heads10.items():
        m = metrics(W, b, quick=True)
        results["orig10"][name] = {p: {k: m["per_phrase"][p]["all"].get(k) for k in ("auroc_first_sib", "top10_at_sib", "auroc_cat_sib", "top10")}
                                   for p in ("New Hampshire", "John Adams", "Connecticut", "Rhode Island")}
        print(f"   {name}: {results['orig10'][name]}", flush=True)

    os.makedirs(f"{V}/results", exist_ok=True)
    json.dump(results, open(f"{V}/results/bench.json", "w"), indent=1)
    vol.commit()
    return results


@app.local_entrypoint()
def main(stage: str = "all"):
    out_dir = os.path.join(here, "results"); os.makedirs(out_dir, exist_ok=True)
    if stage in ("mine", "all"):
        for s in mine.map(range(15)):
            print({k: s[k] for k in ("docs", "secs")}, s["counts"])
    if stage in ("merge", "all"):
        merge.remote()
    if stage in ("collect", "all"):
        collect.remote()
    if stage in ("eval", "all"):
        res = evaluate.remote()
        json.dump(res, open(os.path.join(out_dir, "bench.json"), "w"), indent=1)
        print("wrote", os.path.join(out_dir, "bench.json"))
