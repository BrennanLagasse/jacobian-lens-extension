"""Per-layer pass@10 chart on Qwen3.5-9B: R-lens / logit lens / J-lens (as in the R-lens post) plus our
fitted phrase rows ("J-lens + phrase rows", "R-lens + phrase rows").

Lenses: camilablank/workspace-lenses qwen3.5-9b/{j-lens,r-lens}/lens.pt (post-trained Qwen/Qwen3.5-9B, target 30).
Prompt families (50 each, chart_prompts/*.json): multihop, multilingual, association, typo, poetry (the post's
five, authored to its examples) + multihop_mt (two-hop questions whose intermediate is a multi-token entity).
pass@10 at layer l = fraction of items whose intermediate is in the top-10 of the lens readout at the readout
position. 'std' = first token of any alias in the real-vocab top-10 (the post's rule). 'ext' = top-10 over the
vocabulary extended with fitted rows for every multi-token alias: hit if the post's rule fires there or the item's
phrase row is in it (so 'ext' >= 'std' up to the few real tokens the rows displace). 'rows' = hit via a phrase row only. multihop / multihop_mt / multilingual keep only items the model
answers correctly (greedy, 8 tokens).

Stages:  modal run jlens_chart.py --stage check|mine|merge|collect|fit|eval|plot|all
"""

import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import modal

APP = "jlens-chart"
V = "/vol"
C = f"{V}/chart"
MODEL = "Qwen/Qwen3.5-9B"
LENS_REPO = "camilablank/workspace-lenses"
LENS_FILES = {"jlens": "qwen3.5-9b/j-lens/lens.pt", "rlens": "qwen3.5-9b/r-lens/lens.pt"}
FAMILIES = ["multihop", "multihop_mt", "multilingual", "association", "typo", "poetry"]
FILTERED = {"multihop", "multihop_mt", "multilingual"}
N_HELD, N_TRAIN, N_NEG_PER_CTX = 40, 300, 8
GENERIC_DOCS, GENERIC_EVAL_DOCS, TOKENS_PER_DOC, SKIP = 1000, 100, 512, 4
L2 = 1e-6
LN_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]   # variant "ln": lens-transported generic states at these layers are extra negatives
LN_DOCS = 100

vol = modal.Volume.from_name("jlens-multitoken", create_if_missing=True)
here = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=5.5", "datasets>=3.0", "huggingface_hub", "accelerate", "numpy", "tqdm", "matplotlib")
    .env({"HF_HOME": f"{V}/hf", "PYTHONPATH": "/pkg", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(os.path.join(here, "jlens"), "/pkg/jlens")
    .add_local_dir(os.path.join(here, "chart_prompts"), "/pkg/chart_prompts")
)
app = modal.App(APP, image=image)
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


def load_prompts():
    return {f: json.load(open(f"/pkg/chart_prompts/{f}.json")) for f in FAMILIES}


def phrase_list():
    return json.load(open("/pkg/chart_prompts/phrases.json"))


def phrase_re(phrases):
    return re.compile(r"\b(" + "|".join(sorted(map(re.escape, phrases), key=len, reverse=True)) + r")\b", re.I)


def load_lens(name):
    """camilablank lens.pt -> jlens.JacobianLens (J may be a dict or a stacked tensor; provenance dict present)."""
    import torch
    import jlens
    from huggingface_hub import hf_hub_download

    ck = torch.load(hf_hub_download(LENS_REPO, LENS_FILES[name]), map_location="cpu", weights_only=False)
    J = ck["J"]
    if not isinstance(J, dict):
        J = {int(l): J[i] for i, l in enumerate(ck["source_layers"])}
    return jlens.JacobianLens(jacobians=J, n_prompts=ck.get("n_prompts", 0), d_model=ck["d_model"]), ck.get("provenance")


@app.function(volumes={V: vol}, timeout=1800)
def check():
    import torch
    for name in LENS_FILES:
        lens, prov = load_lens(name)
        l0 = lens.source_layers[0]
        print(name, lens, "| J[l0] norm", float(lens.jacobians[l0].norm()), "| provenance:", {k: str(v)[:80] for k, v in (prov or {}).items()} if isinstance(prov, dict) else prov, flush=True)
    print("phrases", len(phrase_list()), {f: len(v) for f, v in load_prompts().items()})


# ----------------------------------------------------------------------------- mine
@app.function(volumes={V: vol}, timeout=3 * 3600, cpu=2)
def mine(shard: int, n_shards: int = 15, quota: int = 250):
    from datasets import load_dataset

    phrases = phrase_list()
    RE = phrase_re(phrases)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True).shard(num_shards=n_shards, index=shard)
    counts, occ, out, docs, chars, t0 = Counter(), Counter(), [], 0, 0, time.time()
    for doc in ds:
        docs += 1
        text = doc.get("text") or ""
        chars += len(text)
        if not RE.search(text):
            continue
        for m in RE.finditer(text):
            occ[m.group(0).lower()] += 1
        if all(counts[p.lower()] >= quota for p in phrases):
            continue
        sents = _SENT_RE.split(re.sub(r"\s+", " ", text.replace("\n", " ")).strip())
        for i, s in enumerate(sents):
            m = RE.search(s)
            if not m or counts[m.group(0).lower()] >= quota:
                continue
            counts[m.group(0).lower()] += 1
            out.append({"phrase": m.group(0).lower(), "context": " ".join(sents[max(0, i - 2): i + 1]).strip()})
        if docs % 200_000 == 0:
            print(f"[shard {shard}] {docs:,} docs {time.time()-t0:.0f}s min={min(counts[p.lower()] for p in phrases)}", flush=True)
    os.makedirs(f"{C}/mine", exist_ok=True)
    with open(f"{C}/mine/shard_{shard}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"docs": docs, "chars": chars, "occ": dict(occ)}, open(f"{C}/mine/shard_{shard}.stats.json", "w"))
    vol.commit()
    return {"docs": docs, "min_count": min(counts[p.lower()] for p in phrases)}


@app.function(volumes={V: vol}, timeout=1800)
def merge():
    by, stats = defaultdict(dict), {"docs": 0, "chars": 0, "occ": Counter()}
    for fn in sorted(os.listdir(f"{C}/mine")):
        if fn.endswith(".stats.json"):
            s = json.load(open(f"{C}/mine/{fn}"))
            stats["docs"] += s["docs"]; stats["chars"] += s["chars"]; stats["occ"].update(s["occ"])
        else:
            for line in open(f"{C}/mine/{fn}"):
                r = json.loads(line); by[r["phrase"]].setdefault(r["context"], r)
    rng, data, summary = random.Random(1), {}, {}
    for p in phrase_list():
        rs = list(by[p.lower()].values()); rng.shuffle(rs)
        data[p] = [dict(r, split="held") for r in rs[:N_HELD]] + [dict(r, split="train") for r in rs[N_HELD:N_HELD + N_TRAIN]]
        summary[p] = len(rs)
    stats["occ"] = dict(stats["occ"])
    json.dump({"data": data, "stats": stats, "summary": summary}, open(f"{C}/contexts.json", "w"), ensure_ascii=False)
    vol.commit()
    print(json.dumps(summary))
    return summary


# ----------------------------------------------------------------------------- collect
def _load(dev="cuda"):
    import torch, transformers, jlens
    tok = transformers.AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    return tok, model, jlens.from_hf(model, tok)


@app.function(volumes={V: vol}, gpu="H100", timeout=4 * 3600, memory=131072)
def collect(variant: str = ""):
    import numpy as np, torch, jlens
    from datasets import load_dataset

    torch.manual_seed(0)
    dev = "cuda"
    tok, model, lm = _load(dev)
    text_mod, W_U, norm = lm._text_module, lm._lm_head.weight, lm._final_norm
    d = W_U.shape[1]
    Js = {}
    if variant in ("ln", "lp"):
        Js = {n: {l: load_lens(n)[0].jacobians[l].to(dev, torch.bfloat16) for l in LN_LAYERS} for n in LENS_FILES}

    @torch.no_grad()
    def real_stats(h, targets=None):
        out = defaultdict(list)
        for s in range(0, len(h), 1024):
            z = (h[s:s + 1024].to(dev, torch.bfloat16) @ W_U.T).float()
            out["lse"].append(torch.logsumexp(z, 1).cpu()); out["top10"].append(z.topk(10, 1).values[:, -1].cpu())
            if targets is not None:
                out["z_t"].append(z.gather(1, targets[s:s + 1024, None].to(dev)).squeeze(1).cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    Cx = json.load(open(f"{C}/contexts.json"))
    phrases = phrase_list()
    rng, exs, skipped = random.Random(0), [], Counter()
    for pi, p in enumerate(phrases):
        pat = re.compile(r"\b" + re.escape(p) + r"\b", re.I)
        for r in Cx["data"][p]:
            enc = tok(r["context"], return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            m = list(pat.finditer(r["context"]))[-1]
            ti = [i for i, (s, e) in enumerate(offs) if e > s and s < m.end() and e > m.start()]
            if not ti or ti[0] == 0 or len(ids) > 384:
                skipped["start_or_long"] += 1; continue
            negs = [i for i in range(len(ids) - 1) if i != ti[0] - 1]; rng.shuffle(negs)
            exs.append({"ids": ids, "pos": ti[0] - 1, "first": ids[ti[0]], "phrase": pi, "split": r["split"],
                        "negs": sorted(negs[:N_NEG_PER_CTX]) if r["split"] == "train" else []})
    print("examples", len(exs), dict(skipped), flush=True)
    P, N, LP = defaultdict(list), defaultdict(list), defaultdict(list)
    exs.sort(key=lambda e: len(e["ids"]))
    with torch.no_grad():
        for b in range(0, len(exs), 32):
            batch = exs[b:b + 32]
            L = max(len(e["ids"]) for e in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long); mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, e in enumerate(batch):
                ids[i, :len(e["ids"])] = torch.tensor(e["ids"]); mask[i, :len(e["ids"])] = 1
            if variant == "lp":
                with jlens.ActivationRecorder(lm.layers, at=LN_LAYERS) as rec:
                    H = text_mod(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).last_hidden_state
                pos = torch.tensor([e["pos"] for e in batch], device=dev); ar = torch.arange(len(batch), device=dev)
                tr = [i for i, e in enumerate(batch) if e["split"] == "train"]
                for l in LN_LAYERS:
                    r_ = rec.activations[l][ar, pos][tr].to(torch.bfloat16)
                    for n in Js:
                        LP["h"].append(norm(r_ @ Js[n][l].T).cpu()); LP["phrase"].extend(batch[i]["phrase"] for i in tr)
            else:
                H = text_mod(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).last_hidden_state
            for i, e in enumerate(batch):
                P["h"].append(H[i, e["pos"]].cpu()); P["first"].append(e["first"]); P["phrase"].append(e["phrase"]); P["split"].append(e["split"])
                if e["negs"]:
                    N["h"].append(H[i, e["negs"]].cpu()); N["target"].extend(e["ids"][j + 1] for j in e["negs"])
    P["h"] = torch.stack(P["h"]); N["h"] = torch.cat(N["h"])
    P["first"], P["phrase"], N["target"] = map(torch.tensor, (P["first"], P["phrase"], N["target"]))
    P["split"] = np.array(P["split"])
    P.update(real_stats(P["h"], P["first"])); N.update(real_stats(N["h"], N["target"]))
    if LP:
        LP["h"] = torch.cat(LP["h"]); LP["phrase"] = torch.tensor(LP["phrase"]); LP.update(real_stats(LP["h"])); print("lens positives", LP["h"].shape, flush=True)
    print("P", P["h"].shape, "N", N["h"].shape, flush=True)

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    G, LN, n_docs, n_chars, n_tok = defaultdict(list), defaultdict(list), 0, 0, 0
    with torch.no_grad():
        for exd in ds:
            t = exd.get("text") or ""
            if not t.strip():
                continue
            full = tok(t, add_special_tokens=False)["input_ids"]; n_chars += len(t); n_tok += len(full)
            ids = torch.tensor([full[:TOKENS_PER_DOC]], device=dev)
            if ids.shape[1] <= SKIP + 2:
                continue
            if variant in ("ln", "lp") and n_docs < LN_DOCS:
                with jlens.ActivationRecorder(lm.layers, at=LN_LAYERS) as rec:
                    H = text_mod(input_ids=ids, use_cache=False).last_hidden_state[0]
                for l in LN_LAYERS:
                    r_ = rec.activations[l][0, SKIP:-1].to(torch.bfloat16)
                    for n in Js:
                        LN["h"].append(norm(r_ @ Js[n][l].T).cpu()); LN["target"].append(ids[0, SKIP + 1:].cpu())
            else:
                H = text_mod(input_ids=ids, use_cache=False).last_hidden_state[0]
            G["h"].append(H[SKIP:-1].cpu()); G["target"].append(ids[0, SKIP + 1:].cpu()); G["doc"].append(torch.full((ids.shape[1] - SKIP - 1,), n_docs))
            n_docs += 1
            if n_docs % 100 == 0:
                print(f"generic {n_docs}/{GENERIC_DOCS}", flush=True)
            if n_docs >= GENERIC_DOCS:
                break
    G = {k: torch.cat(v) for k, v in G.items()}
    G.update(real_stats(G["h"], G["target"]))
    if LN:
        LN = {k: torch.cat(v) for k, v in LN.items()}; LN.update(real_stats(LN["h"], LN["target"])); print("lens negatives", LN["h"].shape, flush=True)
    torch.save({"P": dict(P), "N": dict(N), "G": G, "LN": dict(LN), "LP": dict(LP), "phrases": phrases, "stats": Cx["stats"], "chars_per_tok": n_chars / n_tok, "d": d},
               f"{C}/states{'_' + variant if variant else ''}.pt")
    vol.commit()
    print("saved states", G["h"].shape, flush=True)


# ----------------------------------------------------------------------------- fit
@app.function(volumes={V: vol}, gpu="H100", timeout=2 * 3600, memory=131072)
def fit(variant: str = ""):
    import torch
    dev = "cuda"
    sfx = "_" + variant if variant else ""
    S = torch.load(f"{C}/states{sfx}.pt", weights_only=False)
    P, N, G, phrases, d = S["P"], S["N"], S["G"], S["phrases"], S["d"]
    k = len(phrases)
    total_tokens = S["stats"]["chars"] / S["chars_per_tok"]
    pi_true = torch.tensor([S["stats"]["occ"].get(p.lower(), 0) / total_tokens for p in phrases]).clamp(min=1e-9)
    train = torch.tensor(P["split"] == "train"); gtrain = G["doc"] < GENERIC_DOCS - GENERIC_EVAL_DOCS
    LN = S.get("LN") or {"h": G["h"][:0], "lse": G["lse"][:0], "z_t": G["z_t"][:0]}
    LP = S.get("LP") or {"h": P["h"][:0], "lse": P["lse"][:0], "phrase": P["phrase"][:0]}
    Hs = torch.cat([P["h"][train], LP["h"], N["h"], G["h"][gtrain], LN["h"]]).to(dev).float()
    lse = torch.cat([P["lse"][train], LP["lse"], N["lse"], G["lse"][gtrain], LN["lse"]]).to(dev)
    ph = torch.cat([P["phrase"][train], LP["phrase"], torch.full((len(N["h"]) + int(gtrain.sum()) + len(LN["h"]),), -1)]).to(dev)
    z_t = torch.cat([torch.zeros(int(train.sum()) + len(LP["h"])), N["z_t"], G["z_t"][gtrain], LN["z_t"]]).to(dev)
    pi_train = torch.tensor([(ph == i).sum().item() / len(Hs) for i in range(k)]).clamp(min=1e-9)
    is_pos, pidx = ph >= 0, ph.clamp(min=0)[:, None]
    wts = torch.where(is_pos, (pi_true / pi_train).to(dev)[pidx.squeeze(1)], torch.ones(len(Hs), device=dev)).double(); wsum = wts.sum()
    W = torch.zeros(k, d, device=dev, requires_grad=True)

    def closure():
        opt.zero_grad(); z = Hs @ W.T
        Lz = torch.logsumexp(torch.cat([lse[:, None], z], 1), 1)
        zt = torch.where(is_pos, z.gather(1, pidx).squeeze(1), z_t)
        loss = ((Lz - zt).double() * wts).sum() / wsum + L2 * W.pow(2).sum(); loss.backward(); return loss
    opt = torch.optim.LBFGS([W], lr=1, max_iter=300, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
    opt.step(closure)
    W = W.detach()

    # held-out calibration report
    held = ~train; geval = ~gtrain
    zp, zg = P["h"][held].to(dev).float() @ W.T, G["h"][geval].to(dev).float() @ W.T
    mass = torch.exp(torch.logsumexp(zg, 1) - torch.logsumexp(torch.cat([G["lse"][geval].to(dev)[:, None], zg], 1), 1)).mean().item()
    rep = {"mass_over_prior": mass / pi_true.sum().item(), "n_train_pos": int(train.sum()), "n_generic_train": int(gtrain.sum()), "per_phrase": {}}
    phH, t10 = P["phrase"][held].to(dev), G["top10"][geval].to(dev)
    for i, p in enumerate(phrases):
        own, gen = zp[phH == i, i], zg[:, i]
        auroc = torch.stack([(gen < o).float().mean() for o in own]).mean().item() if len(own) else float("nan")
        rep["per_phrase"][p] = {"auroc": auroc, "generic_top10": (gen > t10).float().mean().item(), "row_norm": W[i].norm().item(), "n_held": int(len(own)), "prior": pi_true[i].item()}
    rep["n_lens_negatives"] = int(len(LN["h"])); rep["n_lens_positives"] = int(len(LP["h"]))
    torch.save({"rows": W.cpu(), "phrases": phrases, "prior": pi_true}, f"{C}/rows{sfx}.pt")
    json.dump(rep, open(f"{C}/fit_report{sfx}.json", "w"), indent=1)
    vol.commit()
    aur = [v["auroc"] for v in rep["per_phrase"].values()]
    print(f"rows fitted: mean AUROC {sum(aur)/len(aur):.3f}, mean generic top-10 {sum(v['generic_top10'] for v in rep['per_phrase'].values())/k:.4f}, mass/prior {rep['mass_over_prior']:.2f}", flush=True)
    for p, v in sorted(rep["per_phrase"].items(), key=lambda kv: kv[1]["auroc"])[:8]:
        print(f"  weakest: {p:<22} auroc {v['auroc']:.3f} gen-top10 {v['generic_top10']:.4f} |row| {v['row_norm']:.2f}", flush=True)


# ----------------------------------------------------------------------------- eval
@app.function(volumes={V: vol}, gpu="H100", timeout=3 * 3600, memory=131072)
def evaluate(variant: str = ""):
    import torch, jlens
    dev = "cuda"
    sfx = "_" + variant if variant else ""
    tok, model, lm = _load(dev)
    W_U, norm, n_layers = lm._lm_head.weight, lm._final_norm, lm.n_layers
    R = torch.load(f"{C}/rows{sfx}.pt", weights_only=False)
    rows, row_of = R["rows"].to(dev, torch.bfloat16), {p.lower(): i for i, p in enumerate(R["phrases"])}
    lenses = {n: load_lens(n)[0] for n in LENS_FILES}
    Js = {n: {l: L.jacobians[l].to(dev, torch.bfloat16) for l in L.source_layers} for n, L in lenses.items()}
    print({n: (L.source_layers[0], L.source_layers[-1]) for n, L in lenses.items()}, "n_layers", n_layers, flush=True)
    V_ = W_U.shape[0]

    def first_ids(a):
        return {tok.encode(s, add_special_tokens=False)[0] for s in (" " + a, " " + a.lower(), " " + a.capitalize(), a)}

    @torch.no_grad()
    def correct(prompt, answers):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
        cont = tok.decode(out[0, ids.shape[1]:]).lower()
        return any(a.lower() in cont for a in answers), cont

    results, layers = {}, list(range(n_layers))
    prompts = load_prompts()
    # matched control pool: the first tokens of the phrases that have rows (same 57 concepts, real-vocab side)
    pool_first = set().union(*(first_ids(a) for its in prompts.values() for it in its for a in it["intermediate"] if a.lower() in row_of))
    for fam, items in prompts.items():
        keep, cont_log = [], []
        for it in items:
            if fam in FILTERED:
                ok, cont = correct(it["prompt"], it["answer"]); cont_log.append((it["prompt"][-40:], cont[:30], ok))
                if not ok:
                    continue
            keep.append(it)
        curves = {f"{ln}_{kind}": [0.0] * n_layers for ln in ("rlens", "jlens", "logit") for kind in ("std", "ext", "rows", "fp", "stdfp")}
        all_rows = set(range(V_, V_ + rows.shape[0]))
        n_mt = 0
        with torch.no_grad():
            for it in keep:
                std_ids = set().union(*(first_ids(a) for a in it["intermediate"]))
                row_ids = {V_ + row_of[a.lower()] for a in it["intermediate"] if a.lower() in row_of}
                n_mt += bool(row_ids)
                ids = lm.encode(it["prompt"])
                with jlens.ActivationRecorder(lm.layers, at=layers) as rec:
                    lm.forward(ids)
                for l in layers:
                    h = rec.activations[l][0, it["readout"]].to(torch.bfloat16)
                    for ln in ("rlens", "jlens", "logit"):
                        hl = h if (ln == "logit" or l not in Js[ln]) else h @ Js[ln][l].T
                        x = norm(hl).float()
                        z = x @ W_U.float().T
                        top = set(z.topk(10).indices.tolist())
                        curves[f"{ln}_std"][l] += bool(top & std_ids)
                        curves[f"{ln}_stdfp"][l] += bool(top & (pool_first - std_ids))  # matched control: a wrong item's first token in the real top-10
                        ze = torch.cat([z, x @ rows.float().T])
                        tope = set(ze.topk(10).indices.tolist())
                        curves[f"{ln}_ext"][l] += bool(tope & (std_ids | row_ids))  # post's rule OR the phrase row, over the extended vocab
                        curves[f"{ln}_rows"][l] += bool(tope & row_ids)
                        curves[f"{ln}_fp"][l] += bool(tope & (all_rows - row_ids))  # control: a *wrong* phrase row in the top-10
        n = len(keep)
        results[fam] = {"n": n, "n_total": len(items), "n_multitoken": n_mt, "curves": {k: [v / max(n, 1) for v in vs] for k, vs in curves.items()}}
        print(f"{fam}: kept {n}/{len(items)} (multi-token intermediates {n_mt}); pass@10 at L{n_layers-2}: " +
              " ".join(f"{k}={results[fam]['curves'][k][n_layers-2]:.2f}" for k in ("rlens_std", "jlens_std", "logit_std", "jlens_ext", "rlens_ext")), flush=True)
        if fam in FILTERED:
            for pr, cont, ok in cont_log[:6]:
                print(f"   {'ok ' if ok else 'X  '}...{pr!r} -> {cont!r}", flush=True)
    json.dump({"model": MODEL, "lens_repo": LENS_REPO, "n_layers": n_layers, "variant": variant, "results": results}, open(f"{C}/pass10{sfx}.json", "w"), indent=1)
    vol.commit()


# ----------------------------------------------------------------------------- plot
@app.function(volumes={V: vol}, timeout=600)
def plot(variant: str = ""):
    sfx = "_" + variant if variant else ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = json.load(open(f"{C}/pass10{sfx}.json"))
    # ours drawn first, thick and translucent, so the standard lines stay visible where they coincide
    series = [("jlens_ext", "J-lens + phrase rows (ours)", "#2ca02c", "-", 3.2, 0.45), ("rlens_ext", "R-lens + phrase rows (ours)", "#2ca02c", "--", 2.2, 0.6),
              ("rlens_std", "R-lens", "#ff7f0e", "-", 1.6, 1), ("logit_std", "Logit lens", "#c71585", "-", 1.6, 1), ("jlens_std", "J-lens", "#1f77b4", "-", 1.6, 1),
              ("jlens_fp", "control: a wrong phrase row in the J-lens top-10", "#7f7f7f", ":", 1.4, 1),
              ("jlens_stdfp", "matched control: a wrong phrase's first token in the J-lens real top-10", "#1f77b4", ":", 1.4, 1)]
    fig, axes = plt.subplots(2, 3, figsize=(14.7, 7.2))
    for ax, fam in zip(axes.flat, FAMILIES):
        r = D["results"][fam]
        for key, label, color, ls, lw, alpha in series:
            ax.plot(range(D["n_layers"]), r["curves"][key], color=color, ls=ls, lw=lw, alpha=alpha, label=label)
        ax.set_title(f"{fam}  (n={r['n']}, multi-token intermediates={r['n_multitoken']})", fontsize=10)
        ax.set_xlabel("layer"); ax.set_ylabel("pass@10"); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.95))
    note = {"ln": "; rows fitted with lens-transported generic states at layers 2-24 as extra negatives",
            "lp": "; rows fitted on lens-transported states at layers 2-24 (pre-phrase as positives, generic as negatives) + final-layer states"}.get(variant, "; phrase rows fitted on final-layer states")
    fig.suptitle(f"{D['model']}  (J/R lenses: {D['lens_repo']}{note})", y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{C}/pass10_chart{sfx}.png", dpi=150); vol.commit()
    return open(f"{C}/pass10_chart{sfx}.png", "rb").read()


@app.local_entrypoint()
def main(stage: str = "all", variant: str = ""):
    if stage == "check":
        check.remote(); return
    if stage in ("mine", "all"):
        print(list(mine.map(range(15))))
    if stage in ("merge", "all"):
        merge.remote()
    if stage in ("collect", "all"):
        collect.remote(variant)
    if stage in ("fit", "all"):
        fit.remote(variant)
    if stage in ("eval", "all"):
        evaluate.remote(variant)
    if stage in ("plot", "all"):
        png = plot.remote(variant)
        out = os.path.join(here, "results", f"pass10_chart{'_' + variant if variant else ''}.png")
        open(out, "wb").write(png); print("wrote", out)
