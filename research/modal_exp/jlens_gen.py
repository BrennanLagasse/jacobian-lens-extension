"""Generalisation study for phrase heads: beyond proper nouns (Modal).

Axes (run one at a time, or 'joint' at the end):
  nouns    compositional noun phrases with first-token sibling groups (ice cream / ice age / ice hockey)
  verbs    verb phrases and idioms with first-token sibling groups (take into account / take place / ...)
  abstract abstract concepts defined by surface-form sets (betrayal: betray, betrayed, treachery, ...)
  xling    the same concepts in English / Spanish / French / Chinese (FineWeb-2), cross-lingual transfer
  events   numbers, dates and multi-word events with jagged tokenisation
  joint    all rows of every axis fitted in one extended softmax vs separately (the confound, last)

Per axis: mine contexts (regex, 15 FineWeb shards), collect states (pre-phrase 'before' position, the
positions inside the phrase, R-lens / J-lens / logit-lens transported states at 6 layers for held-out
contexts and for 20k generic positions), fit the bias-free logistic head (extended softmax, positives
weighted to the corpus prior) and a presence probe (binary logistic on before+inside positions), and
score with ROC-style metrics: TPR at fixed FPR vs generic text and vs siblings, FPR at fixed TPR,
per-layer TPR at FPR 1e-2 under each lens, earliest layer at which TPR@FPR=1e-2 reaches 0.5.

Generic negatives / evaluation positions are reused from the benchmark (states/bench.pt: G, E).
"""

import json
import math
import os
import re
import time
import types
from collections import Counter, defaultdict

import modal

APP = "jlens-gen"
V = "/vol"
MODEL = "Qwen/Qwen3.5-9B-Base"
TARGET_LAYER = 30
N_HELD = 200
N_TRAIN = 1200
N_NEG_PER_CTX = 8
LENS_LAYERS = [8, 12, 16, 20, 24, 28]
N_GENERIC_LENS_DOCS = 60           # generic docs whose positions get lens states (~20k positions)
FPRS = [1e-4, 1e-3, 1e-2]

AXES = {
    "nouns": {"dataset": "fineweb", "boundary": "word", "groups": {
        "ice": ["ice cream", "ice age", "ice hockey"], "credit": ["credit card", "credit score", "credit union"],
        "machine": ["machine learning", "machine gun"], "social": ["social media", "social security", "social network"],
        "high": ["high school", "high court", "high speed"], "human": ["human rights", "human resources", "human nature"],
        "solar": ["solar system", "solar panel", "solar energy"], "black": ["black hole", "black market", "black pepper"],
        "prime": ["prime minister", "prime number", "prime time"], "stock": ["stock market", "stock price", "stock exchange"],
        "civil": ["civil war", "civil rights", "civil engineering"], "middle": ["middle class", "middle school", "middle east"],
        "hard": ["hard drive", "hard disk", "hard work"], "real": ["real estate", "real time", "real world"],
        "climate": ["climate change", "climate crisis"], "health": ["health care", "health insurance"]}},
    "verbs": {"dataset": "fineweb", "boundary": "word", "groups": {
        "take": ["take into account", "take place", "take advantage", "take care"], "kick": ["kick the bucket", "kick off"],
        "in": ["in spite of", "in addition to", "in front of", "in terms of"], "make": ["make sense", "make sure", "make up"],
        "give": ["give up", "give rise to", "give birth"], "look": ["look forward to", "look after", "look up"],
        "as": ["as well as", "as soon as", "as long as"], "on": ["on behalf of", "on the other hand", "on top of"],
        "come": ["come up with", "come across", "come to terms"], "break": ["break down", "break even", "break the ice"],
        "at": ["at the end of the day", "at least", "at the same time"], "get": ["get rid of", "get along", "get away with"],
        "put": ["put up with", "put forward", "put off"], "by": ["by and large", "by the way", "by means of"]}},
    "abstract": {"dataset": "fineweb", "boundary": "word", "groups": {
        # each phrase = concept, its surface forms are mined together (last form occurrence in the context is the target)
        "emotion": ["betrayal", "regret", "forgiveness", "jealousy", "nostalgia", "gratitude", "loneliness", "curiosity"],
        "character": ["irony", "hypocrisy", "ambition"], "economy": ["bankruptcy", "inflation", "negotiation", "democracy"],
        "science": ["evolution", "photosynthesis", "gravity", "entropy", "recursion"]},
        "forms": {
        "betrayal": ["betrayal", "betrayed", "betray", "betrays", "treachery", "double-crossed", "backstabbed"],
        "irony": ["irony", "ironic", "ironically"], "regret": ["regret", "regretted", "regrets", "remorse", "regretful"],
        "negotiation": ["negotiation", "negotiations", "negotiate", "negotiated", "negotiating", "bargaining"],
        "forgiveness": ["forgiveness", "forgive", "forgave", "forgiven", "pardoned", "absolution"],
        "jealousy": ["jealousy", "jealous", "envy", "envious"], "curiosity": ["curiosity", "curious", "inquisitive"],
        "nostalgia": ["nostalgia", "nostalgic", "wistful"], "hypocrisy": ["hypocrisy", "hypocrite", "hypocritical", "hypocrites"],
        "ambition": ["ambition", "ambitious", "ambitions"], "gratitude": ["gratitude", "grateful", "thankful", "thankfulness"],
        "loneliness": ["loneliness", "lonely", "loneliest", "solitude"], "bankruptcy": ["bankruptcy", "bankrupt", "insolvency", "insolvent"],
        "inflation": ["inflation", "inflationary"], "democracy": ["democracy", "democratic", "democracies"],
        "evolution": ["evolution", "evolutionary", "evolved", "evolve"], "photosynthesis": ["photosynthesis", "photosynthetic"],
        "gravity": ["gravity", "gravitational", "gravitation"], "entropy": ["entropy", "entropic"], "recursion": ["recursion", "recursive", "recursively"]}},
    "events": {"dataset": "fineweb", "boundary": "word", "groups": {
        "french": ["French Revolution", "French Riviera"], "industrial": ["Industrial Revolution", "industrial park"],
        "world": ["World War II", "World War I", "World Cup", "World Series"], "cold": ["Cold War", "cold front"],
        "great": ["Great Depression", "Great Wall", "Great Barrier Reef", "Great Recession"], "september": ["September 11", "September 2008"],
        "july": ["July 4, 1776", "July 20, 1969"], "y1984": ["1984", "1989"], "y2008": ["2008 financial crisis", "2008 Olympics"],
        "civilwar": ["American Civil War", "Spanish Civil War"], "moon": ["moon landing", "moon phase"], "berlin": ["Berlin Wall", "Berlin Marathon"]}},
}
# cross-lingual: the same 12 concepts in four languages; each language is mined from its own corpus
XLING_CONCEPTS = {
    "ice cream": {"eng": ["ice cream"], "spa": ["helado"], "fra": ["glace à la vanille", "crème glacée"], "zho": ["冰淇淋", "冰激凌"]},
    "climate change": {"eng": ["climate change"], "spa": ["cambio climático"], "fra": ["changement climatique"], "zho": ["气候变化"]},
    "credit card": {"eng": ["credit card"], "spa": ["tarjeta de crédito"], "fra": ["carte de crédit"], "zho": ["信用卡"]},
    "human rights": {"eng": ["human rights"], "spa": ["derechos humanos"], "fra": ["droits de l'homme", "droits humains"], "zho": ["人权"]},
    "solar system": {"eng": ["solar system"], "spa": ["sistema solar"], "fra": ["système solaire"], "zho": ["太阳系"]},
    "black hole": {"eng": ["black hole"], "spa": ["agujero negro"], "fra": ["trou noir"], "zho": ["黑洞"]},
    "prime minister": {"eng": ["prime minister"], "spa": ["primer ministro"], "fra": ["premier ministre"], "zho": ["首相", "总理"]},
    "stock market": {"eng": ["stock market"], "spa": ["bolsa de valores", "mercado de valores"], "fra": ["marché boursier", "bourse"], "zho": ["股市", "股票市场"]},
    "civil war": {"eng": ["civil war"], "spa": ["guerra civil"], "fra": ["guerre civile"], "zho": ["内战"]},
    "middle class": {"eng": ["middle class"], "spa": ["clase media"], "fra": ["classe moyenne"], "zho": ["中产阶级"]},
    "real estate": {"eng": ["real estate"], "spa": ["bienes raíces", "sector inmobiliario"], "fra": ["immobilier"], "zho": ["房地产"]},
    "health insurance": {"eng": ["health insurance"], "spa": ["seguro médico", "seguro de salud"], "fra": ["assurance maladie"], "zho": ["医疗保险", "医保"]},
}
XLING_LANGS = {"eng": ("fineweb", None), "spa": ("fineweb-2", "spa_Latn"), "fra": ("fineweb-2", "fra_Latn"), "zho": ("fineweb-2", "cmn_Hani")}

vol = modal.Volume.from_name("jlens-multitoken", create_if_missing=True)
here = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=5.5", "datasets>=3.0", "huggingface_hub", "accelerate", "numpy", "tqdm", "scikit-learn")
    .env({"HF_HOME": f"{V}/hf", "PYTHONPATH": "/pkg", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(os.path.join(here, "jlens"), "/pkg/jlens")
)
app = modal.App(APP, image=image)
_SENT_RE = re.compile(r'(?:(?<=[.!?])\s+(?=[A-Z0-9"\'¿¡«]))|(?<=[。！？])')


def axis_phrases(axis):
    """-> list of (phrase_key, group, forms, lang)."""
    out = []
    if axis == "xling":
        for concept, langs in XLING_CONCEPTS.items():
            for lang, forms in langs.items():
                out.append((f"{concept}|{lang}", concept, forms, lang))
        return out
    spec = AXES[axis]
    for g, phrases in spec["groups"].items():
        for p in phrases:
            out.append((p, g, spec.get("forms", {}).get(p, [p]), "eng"))
    return out


def phrase_re(forms, boundary):
    alt = "|".join(sorted(map(re.escape, forms), key=len, reverse=True))
    return re.compile((r"\b(" + alt + r")\b") if boundary == "word" else "(" + alt + ")", re.I)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return _SENT_RE.split(text) if text else []


# ----------------------------------------------------------------------------- mine
@app.function(volumes={V: vol}, timeout=3 * 3600, cpu=2)
def mine(shard: int, axis: str, lang: str = "eng", n_shards: int = 15, quota: int = 250, max_docs: int = 1_200_000):
    from datasets import load_dataset

    items = [it for it in axis_phrases(axis) if it[3] == lang]
    boundary = "substr" if lang == "zho" else "word"
    form_res = [(re.compile(re.escape(f), re.I), key) for key, g, forms, _ in items for f in forms]
    _cache = {}

    def keyof_text(t):
        if t not in _cache:
            _cache[t] = next((key for fr, key in form_res if fr.fullmatch(t)), None)
        return _cache[t]
    RE = phrase_re([f for _, _, forms, _ in items for f in forms], boundary)
    ds_name, cfg = ("fineweb", None) if lang == "eng" else XLING_LANGS[lang]
    if ds_name == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-2", name=cfg, split="train", streaming=True)
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
        if not RE.search(text):
            continue
        for m in RE.finditer(text):
            k_ = keyof_text(m.group(0))
            if k_ is not None:
                occ[k_] += 1
        if all(counts[k] >= quota for k, *_ in items):
            continue
        sents = split_sentences(text)
        for i, s in enumerate(sents):
            m = RE.search(s)
            if not m:
                continue
            key = keyof_text(m.group(0))
            if key is None or counts[key] >= quota:
                continue
            counts[key] += 1
            out.append({"phrase": key, "matched": m.group(0), "context": " ".join(sents[max(0, i - 2): i + 1]).strip(), "shard": shard})
        if docs % 200_000 == 0:
            print(f"[{axis}/{lang} shard {shard}] {docs:,} docs {time.time()-t0:.0f}s min count {min(counts[k] for k, *_ in items) if items else 0}", flush=True)
    d = f"{V}/data/gen_{axis}_{lang}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/shard_{shard}.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"docs": docs, "chars": chars, "occ": dict(occ), "counts": dict(counts), "secs": time.time() - t0}, open(f"{d}/shard_{shard}.stats.json", "w"))
    vol.commit()
    return {"docs": docs, "counts": dict(counts)}


@app.function(volumes={V: vol}, timeout=1800)
def merge(axis: str):
    import random
    langs = sorted({it[3] for it in axis_phrases(axis)})
    by, stats = defaultdict(dict), {}
    for lang in langs:
        d = f"{V}/data/gen_{axis}_{lang}"
        st = {"docs": 0, "chars": 0, "occ": Counter()}
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".stats.json"):
                s = json.load(open(f"{d}/{fn}")); st["docs"] += s["docs"]; st["chars"] += s["chars"]; st["occ"].update(s["occ"])
            elif fn.endswith(".jsonl"):
                for line in open(f"{d}/{fn}"):
                    r = json.loads(line); by[r["phrase"]].setdefault(r["context"], r)
        st["occ"] = dict(st["occ"]); stats[lang] = st
    rng = random.Random(2)
    data, summary = {}, {}
    for key, g, forms, lang in axis_phrases(axis):
        rs = list(by[key].values()); rng.shuffle(rs)
        held, train = rs[:N_HELD], rs[N_HELD:N_HELD + N_TRAIN]
        for i, r in enumerate(held):
            r["split"], r["order"] = "held", i
        for i, r in enumerate(train):
            r["split"], r["order"] = "train", i
        data[key] = held + train
        summary[key] = {"unique": len(rs), "held": len(held), "train": len(train), "group": g, "lang": lang}
    json.dump({"data": data, "stats": stats, "summary": summary}, open(f"{V}/data/gen_{axis}.json", "w"), ensure_ascii=False)
    vol.commit()
    print(json.dumps({k: (v["unique"], v["train"]) for k, v in summary.items()}, ensure_ascii=False))
    return summary


# ----------------------------------------------------------------------------- collect
@app.function(volumes={V: vol}, gpu="H100", timeout=4 * 3600, memory=131072)
def collect(axis: str):
    import numpy as np
    import torch
    import transformers
    from datasets import load_dataset
    import jlens

    torch.manual_seed(0)
    dev = "cuda"
    tok = transformers.AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    lm = jlens.from_hf(model, tok)
    text_mod, W_U, norm = lm._text_module, lm._lm_head.weight, lm._final_norm
    Vsz, d = W_U.shape
    vol.reload()
    lenses = {n: jlens.JacobianLens.load(f"{V}/lenses/base_{n}_t{TARGET_LAYER}.pt") for n in ("jlens", "rlens")}
    Js = {n: {l: L.jacobians[l].to(dev, torch.bfloat16) for l in L.source_layers} for n, L in lenses.items()}
    LENSES = ("rlens", "jlens", "logit")

    @torch.no_grad()
    def fwd_rec(ids, mask):
        with jlens.ActivationRecorder(lm.layers, at=LENS_LAYERS) as rec:
            H = text_mod(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        return H, {l: rec.activations[l] for l in LENS_LAYERS}

    def transported(res_l, l, name):
        return norm(res_l if (name == "logit" or l not in Js[name]) else res_l @ Js[name][l].T)

    @torch.no_grad()
    def real_stats(h, targets=None):
        out = defaultdict(list)
        for s in range(0, h.shape[0], 1024):
            z = (h[s:s + 1024].to(torch.bfloat16) @ W_U.T).float()
            out["lse"].append(torch.logsumexp(z, 1).cpu()); out["top10"].append(z.topk(10, 1).values[:, 9].cpu())
            if targets is not None:
                out["z_t"].append(z.gather(1, targets[s:s + 1024, None].to(dev)).squeeze(1).cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    C = json.load(open(f"{V}/data/gen_{axis}.json"))
    items = axis_phrases(axis)
    keys = [it[0] for it in items]
    kidx = {k: i for i, k in enumerate(keys)}
    import random
    rng = random.Random(0)
    ex, skipped = [], Counter()
    for key, g, forms, lang in items:
        boundary = "substr" if lang == "zho" else "word"
        pat = phrase_re(forms, boundary)
        for r in C["data"][key]:
            enc = tok(r["context"], return_offsets_mapping=True, add_special_tokens=False)
            ids_, offs = enc["input_ids"], enc["offset_mapping"]
            ms = list(pat.finditer(r["context"]))
            if not ms:
                skipped["nomatch"] += 1; continue
            m = ms[-1]
            ti = [i for i, (s_, e_) in enumerate(offs) if e_ > s_ and s_ < m.end() and e_ > m.start()]
            if not ti or ti[0] == 0 or len(ids_) > 384:
                skipped["start_or_long"] += 1; continue
            t0_, t1_ = ti[0], ti[-1]
            negs = [i for i in range(len(ids_) - 1) if i < t0_ - 1 or i > t1_]; rng.shuffle(negs)
            ex.append({"ids": ids_, "pos": t0_ - 1, "t0": t0_, "t1": t1_, "first": ids_[t0_], "phrase": kidx[key], "split": r["split"],
                       "order": r["order"], "n_occ": len(ms), "negs": sorted(negs[:N_NEG_PER_CTX]) if r["split"] == "train" else []})
    print(axis, "examples", len(ex), "skipped", dict(skipped), flush=True)
    P, N, IN, HL, POSP = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for split in ("train", "held"):
        exs = sorted([e for e in ex if e["split"] == split], key=lambda e: len(e["ids"]))
        bs = 16
        for b in range(0, len(exs), bs):
            batch = exs[b:b + bs]
            L = max(len(e["ids"]) for e in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long); mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, e in enumerate(batch):
                ids[i, :len(e["ids"])] = torch.tensor(e["ids"]); mask[i, :len(e["ids"])] = 1
            H, res = fwd_rec(ids.to(dev), mask.to(dev))
            for i, e in enumerate(batch):
                P["h"].append(H[i, e["pos"]].cpu())
                for k in ("phrase", "split", "order", "first", "n_occ"):
                    P[k].append(e[k])
                # inside positions (t0..t1): the phrase is 'active'
                IN["h"].append(H[i, e["t0"]:e["t1"] + 1].cpu()); IN["phrase"].extend([e["phrase"]] * (e["t1"] - e["t0"] + 1))
                IN["split"].extend([split] * (e["t1"] - e["t0"] + 1)); IN["order"].extend([e["order"]] * (e["t1"] - e["t0"] + 1))
                if e["negs"]:
                    N["h"].append(H[i, e["negs"]].cpu()); N["target"].extend(e["ids"][j + 1] for j in e["negs"])
                    N["phrase"].extend([e["phrase"]] * len(e["negs"])); N["order"].extend([e["order"]] * len(e["negs"]))
                if split == "held":
                    # lens states at the 'before' position, plus a position profile before-1 .. t1+1 at the final layer
                    for name in LENSES:
                        HL[name].append(torch.stack([transported(res[l][i, e["pos"]], l, name) for l in LENS_LAYERS]).cpu())
                    lo, hi = max(0, e["pos"] - 1), min(len(e["ids"]) - 1, e["t1"] + 1)
                    POSP["h"].append(H[i, lo:hi + 1].cpu()); POSP["offset"].extend(range(lo - e["pos"], hi - e["pos"] + 1))
                    POSP["phrase"].extend([e["phrase"]] * (hi - lo + 1)); POSP["rel"].extend([("before" if j < e["t0"] else ("inside" if j <= e["t1"] else "after")) for j in range(lo, hi + 1)])
            if (b // bs) % 100 == 0:
                print(f"  {split} {b}/{len(exs)}", flush=True)
    P["h"] = torch.stack(P["h"]); N["h"] = torch.cat(N["h"]); IN["h"] = torch.cat(IN["h"]); POSP["h"] = torch.cat(POSP["h"])
    for D, ks in ((P, ("phrase", "order", "first", "n_occ")), (N, ("target", "phrase", "order")), (IN, ("phrase", "order")), (POSP, ("offset", "phrase"))):
        for k in ks:
            D[k] = torch.tensor(D[k])
    P["split"] = np.array(P["split"]); IN["split"] = np.array(IN["split"]); POSP["rel"] = np.array(POSP["rel"])
    for name in LENSES:
        HL[name] = torch.stack(HL[name])  # [nH, nL, d]
    held = torch.tensor(P["split"] == "held")
    st = real_stats(P["h"].to(dev), P["first"]); P.update({"lse": st["lse"], "top10": st["top10"], "z_first": st["z_t"]})
    st = real_stats(N["h"].to(dev), N["target"]); N.update({"lse": st["lse"], "z_t": st["z_t"]})
    nH, nL = HL["rlens"].shape[:2]
    for name in LENSES:
        st = real_stats(HL[name].reshape(-1, d).to(dev), P["first"][held].repeat_interleave(nL))
        HL[f"{name}_top10"] = st["top10"].reshape(nH, nL); HL[f"{name}_zfirst"] = st["z_t"].reshape(nH, nL)
    print("P", P["h"].shape, "held", int(held.sum()), "N", N["h"].shape, "IN", IN["h"].shape, "POSP", POSP["h"].shape, flush=True)

    # generic lens states (shared across axes; computed once)
    gpath = f"{V}/states/generic_lens.pt"
    if not os.path.exists(gpath):
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True).shuffle(seed=7, buffer_size=10_000)
        GL = defaultdict(list); n_docs = 0
        for exd in ds:
            t = exd.get("text") or ""
            if not t.strip():
                continue
            enc = tok([t], return_tensors="pt", truncation=True, max_length=384)
            ids = enc.input_ids.to(dev)
            H, res = fwd_rec(ids, torch.ones_like(ids))
            Lq = ids.shape[1]
            if Lq <= 6:
                continue
            for name in LENSES:
                GL[name].append(torch.stack([transported(res[l][0, 4:Lq - 1], l, name) for l in LENS_LAYERS], 1).cpu())  # [n, nL, d]
            GL["h"].append(H[0, 4:Lq - 1].cpu()); GL["target"].append(ids[0, 5:Lq].cpu())
            n_docs += 1
            if n_docs >= N_GENERIC_LENS_DOCS:
                break
        for name in LENSES:
            GL[name] = torch.cat(GL[name])
            st = real_stats(GL[name].reshape(-1, d).to(dev)); GL[f"{name}_top10"] = st["top10"].reshape(GL[name].shape[0], nL)
        GL["h"] = torch.cat(GL["h"]); GL["target"] = torch.cat(GL["target"])
        torch.save(dict(GL), gpath); vol.commit()
        print("generic lens states", GL["rlens"].shape, flush=True)

    os.makedirs(f"{V}/states", exist_ok=True)
    torch.save({"P": dict(P), "N": dict(N), "IN": dict(IN), "POSP": dict(POSP), "HL": dict(HL), "keys": keys, "items": items,
                "stats": C["stats"], "summary": C["summary"], "d": d, "generic_chars_per_tok": None}, f"{V}/states/gen_{axis}.pt")
    vol.commit()
    print("saved", flush=True)


# ----------------------------------------------------------------------------- eval
@app.function(volumes={V: vol}, gpu="A100-80GB", timeout=3 * 3600, memory=131072)
def evaluate(axis: str, joint_axes: str = ""):
    """joint_axes: comma-separated list -> fit all their rows in one extended softmax (the 'joint' confound)."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, roc_curve

    dev = "cuda"
    vol.reload()
    B = torch.load(f"{V}/states/bench.pt", weights_only=False)
    G, E = B["G"], B["E"]
    chars_per_tok = B["generic_chars"] / B["generic_tokens"]
    GL = torch.load(f"{V}/states/generic_lens.pt", weights_only=False)
    axes = [a for a in joint_axes.split(",") if a] or [axis]
    parts = [torch.load(f"{V}/states/gen_{a}.pt", weights_only=False) for a in axes]
    # concatenate rows across axes (offset phrase ids)
    keys, P, N, IN, POSP, HL, stats, summary = [], defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), {}, {}
    off = 0
    for a, S in zip(axes, parts):
        nk = len(S["keys"]); keys += [f"{k}" if len(axes) == 1 else f"{k}@{a}" for k in S["keys"]]
        for D, T in ((S["P"], P), (S["N"], N), (S["IN"], IN), (S["POSP"], POSP)):
            for k, v in D.items():
                T[k].append(v + off if k == "phrase" else v)
        for k, v in S["HL"].items():
            HL[k].append(v)
        stats[a] = S["stats"]; summary.update({(k if len(axes) == 1 else f"{k}@{a}"): v for k, v in S["summary"].items()})
        off += nk
    cat = lambda xs: (np.concatenate(xs) if isinstance(xs[0], np.ndarray) else torch.cat(xs))
    P, N, IN, POSP, HL = ({k: cat(v) for k, v in D.items()} for D in (P, N, IN, POSP, HL))
    items = [it for S in parts for it in S["items"]]
    n_phr, d = len(keys), parts[0]["d"]
    held = torch.tensor(P["split"] == "held"); train = ~held
    phH = P["phrase"][held]
    # priors: occurrences per token in the mined corpus of the phrase's language
    pi = []
    for (key, g, forms, lang), a in [(it, a) for a, S in zip(axes, parts) for it in S["items"]]:
        st = stats[a][lang]; pi.append(st["occ"].get(key, 0) / (st["chars"] / chars_per_tok))
    pi_true = torch.tensor(pi).clamp(min=1e-9)
    hE, lseE, ztE, top10E = E["h"].to(dev).float(), E["lse"].to(dev), E["z_t"].to(dev), E["top_v"][:, 9].to(dev)
    hH = P["h"][held].to(dev).float()

    # ---- heads --------------------------------------------------------------------------------
    def fit_lr(lam=1e-6):
        idx_pos = torch.where(train)[0]; idx_neg = torch.where(N["order"] >= 0)[0]
        Hs = torch.cat([P["h"][idx_pos], N["h"][idx_neg], G["h"]]).to(dev).float()
        lse = torch.cat([P["lse"][idx_pos], N["lse"][idx_neg], G["lse"]]).to(dev)
        ph = torch.cat([P["phrase"][idx_pos], torch.full((len(idx_neg) + len(G["h"]),), -1)]).to(dev)
        z_t = torch.cat([torch.zeros(len(idx_pos)), N["z_t"][idx_neg], G["z_t"]]).to(dev)
        n_tot = len(Hs); pi_train = torch.tensor([(ph == i).sum().item() / n_tot for i in range(n_phr)]).clamp(min=1e-9)
        is_pos = ph >= 0; pidx = ph.clamp(min=0)[:, None]
        wts = torch.where(is_pos, (pi_true / pi_train).to(dev)[pidx.squeeze(1)], torch.ones(n_tot, device=dev)).double(); wsum = wts.sum()
        W = torch.zeros(n_phr, d, device=dev, requires_grad=True)

        def closure():
            opt.zero_grad(); z = Hs @ W.T
            Lz = torch.logsumexp(torch.cat([lse[:, None], z], 1), 1)
            zt = torch.where(is_pos, z.gather(1, pidx).squeeze(1), z_t)
            loss = ((Lz - zt).double() * wts).sum() / wsum + lam * W.pow(2).sum(); loss.backward(); return loss
        opt = torch.optim.LBFGS([W], lr=1, max_iter=300, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
        opt.step(closure)
        del Hs
        return W.detach()

    def fit_probe(lam=1e-4):
        """Presence probe: one binary logistic row per phrase, positives = before + inside positions of the phrase
        (weighted to the corpus prior), negatives = every other stored position. No softmax competition."""
        inside_tr = torch.tensor(IN["split"] == "train")
        Hs = torch.cat([P["h"][train], IN["h"][inside_tr], N["h"], G["h"][:200000]]).to(dev).float()
        ph = torch.cat([P["phrase"][train], IN["phrase"][inside_tr], torch.full((len(N["h"]) + 200000,), -1)]).to(dev)
        n_tot = len(Hs)
        W = torch.zeros(n_phr, d, device=dev, requires_grad=True); b = torch.zeros(n_phr, device=dev, requires_grad=True)
        Y = torch.nn.functional.one_hot(ph.clamp(min=0), n_phr).double() * (ph >= 0)[:, None]
        pi_train = Y.mean(0).clamp(min=1e-9)
        w_pos = (pi_true.to(dev).double() / pi_train)  # per-phrase positive weight
        Wt = torch.where(Y > 0, w_pos[None, :], torch.ones_like(Y))

        def closure():
            opt.zero_grad(); z = (Hs @ W.T + b).double()
            loss = (Wt * torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none")).sum() / Wt.sum() + lam * W.pow(2).sum(); loss.backward(); return loss
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
        opt.step(closure)
        del Hs
        return W.detach(), b.detach()

    # ---- ROC helpers ----------------------------------------------------------------------------
    def roc_metrics(pos, neg):
        pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; s = np.r_[pos, neg]
        fpr, tpr, thr = roc_curve(y, s)
        out = {"auroc": float(roc_auc_score(y, s)), "n_pos": int(len(pos)), "n_neg": int(len(neg))}
        negs = np.sort(neg)[::-1]
        for f_ in FPRS:
            k = int(np.ceil(f_ * len(neg)))
            if k < 1:
                out[f"tpr@fpr{f_:g}"] = None; continue
            t = negs[k - 1]
            out[f"tpr@fpr{f_:g}"] = float((pos > t).mean())
        for t_ in (0.5, 0.9):
            q = np.quantile(pos, 1 - t_)
            out[f"fpr@tpr{t_:g}"] = float((neg >= q).mean())
        return out

    results = {"axis": axis, "axes": axes, "keys": keys, "summary": summary, "pi_true": dict(zip(keys, pi_true.tolist())),
               "lens_layers": LENS_LAYERS, "n_generic_eval": len(E["h"]), "n_generic_lens": int(GL["rlens"].shape[0]), "heads": {}}
    groups = defaultdict(list)
    for i, (key, g, forms, lang) in enumerate(items):
        groups[g].append(i)
    W_lr = fit_lr()
    W_pr, b_pr = fit_probe()
    print("heads fitted", flush=True)

    with torch.no_grad():
        for hname, W, b in (("lr_nobias", W_lr, torch.zeros(n_phr, device=dev)), ("probe", W_pr, b_pr)):
            zE = (hE @ W.T + b).cpu().numpy(); zH = (hH @ W.T + b).cpu().numpy()
            zIN = (IN["h"][torch.tensor(IN["split"] == "held")].to(dev).float() @ W.T + b).cpu().numpy(); phIN = IN["phrase"][torch.tensor(IN["split"] == "held")].numpy()
            zPP = (POSP["h"].to(dev).float() @ W.T + b).cpu().numpy()
            zGL = {name: torch.einsum("nld,pd->nlp", GL[name].to(dev).float(), W).cpu().numpy() + b.cpu().numpy() for name in ("rlens", "jlens", "logit")}
            zHL = {name: torch.einsum("nld,pd->nlp", HL[name].to(dev).float(), W).cpu().numpy() + b.cpu().numpy() for name in ("rlens", "jlens", "logit")}
            per = {}
            for p, key in enumerate(keys):
                own_t = phH == p; own = own_t.numpy()
                if own.sum() < 10:
                    continue
                r = {"group": items[p][1], "lang": items[p][3], "n_held": int(own.sum())}
                r["vs_generic"] = roc_metrics(zH[own, p], zE[:, p])
                if axis == "xling":
                    lang_p = items[p][3]
                    same_lang_other = [q for q in range(n_phr) if items[q][3] == lang_p and q != p]
                    r["vs_same_language_other_concepts"] = roc_metrics(zH[own, p], zH[np.isin(phH.numpy(), same_lang_other), p])
                sib = [q for q in groups[items[p][1]] if q != p]
                if sib:
                    sibm = np.isin(phH.numpy(), sib)
                    r["vs_sibling"] = roc_metrics(zH[own, p], zH[sibm, p]); r["siblings"] = [keys[q] for q in sib]
                # rank of the phrase token (extended vocab) at own positions, final layer, and the first token's rank
                if hname == "lr_nobias":
                    top10 = P["top10"][held][own_t].numpy()
                    r["own_top10"] = float((zH[own, p] > top10).mean())
                    r["first_tok_top10"] = float((P["z_first"][held][own_t].numpy() >= top10).mean())
                    r["generic_top10"] = float((zE[:, p] > top10E.cpu().numpy()).mean())
                    r["mass_ratio"] = float(np.exp(zE[:, p] - np.logaddexp(lseE.cpu().numpy(), zE[:, p])).mean() / pi_true[p].item())
                # inside positions: does the row stay on while the phrase is being read?  TPR at the generic FPR=1e-2 threshold
                thr = np.quantile(zE[:, p], 0.99)
                r["inside_tpr@fpr1e-2"] = float((zIN[phIN == p, p] > thr).mean()) if (phIN == p).any() else None
                r["position_profile"] = {}
                for offv in sorted(set(POSP["offset"][POSP["phrase"] == p].tolist())):
                    sel = ((POSP["phrase"] == p) & (POSP["offset"] == offv)).numpy()
                    if sel.sum() >= 20:
                        r["position_profile"][str(offv)] = float((zPP[sel, p] > thr).mean())
                # lens layer profiles: TPR at FPR 1e-2 per layer (threshold from generic lens states of the same lens/layer), and top-10
                r["lens"] = {}
                for name in ("rlens", "jlens", "logit"):
                    tprs, top10s, first_top10s = [], [], []
                    for li in range(len(LENS_LAYERS)):
                        thr_l = np.quantile(zGL[name][:, li, p], 0.99)
                        tprs.append(float((zHL[name][own, li, p] > thr_l).mean()))
                        top10s.append(float((zHL[name][own, li, p] > HL[f"{name}_top10"][own_t, li].numpy()).mean()))
                        first_top10s.append(float((HL[f"{name}_zfirst"][own_t, li].numpy() >= HL[f"{name}_top10"][own_t, li].numpy()).mean()))
                    first_layer = next((LENS_LAYERS[i] for i, t in enumerate(tprs) if t >= 0.5), 99)
                    r["lens"][name] = {"tpr@fpr1e-2_by_layer": tprs, "top10_by_layer": top10s, "first_tok_top10_by_layer": first_top10s, "earliest_layer_tpr50": first_layer}
                per[key] = r
            # cross-lingual transfer (row of one language scored at another language's positions)
            xfer = None
            if axis == "xling" and len(axes) == 1:
                xfer = {}
                for p, key in enumerate(keys):
                    concept, lang = key.split("|")
                    for q, key2 in enumerate(keys):
                        c2, lang2 = key2.split("|")
                        if c2 != concept or q == p:
                            continue
                        ownq = (phH == q).numpy()
                        if ownq.sum() < 10:
                            continue
                        thr = np.quantile(zE[:, p], 0.99)
                        # language-matched control: positions of the other concepts in the *target* language
                        others_b = np.isin(phH.numpy(), [r_ for r_, k3 in enumerate(keys) if k3.endswith("|" + lang2) and k3.split("|")[0] != concept])
                        within = roc_metrics(zH[ownq, p], zH[others_b, p])
                        xfer[f"{concept}|{lang}->{lang2}"] = {"tpr@fpr1e-2": float((zH[ownq, p] > thr).mean()), "auroc_vs_generic": roc_metrics(zH[ownq, p], zE[:, p])["auroc"],
                                                              "auroc_within_lang": within["auroc"], "tpr@fpr1e-2_within_lang": within.get("tpr@fpr0.01")}
            # row redundancy
            Wn = torch.nn.functional.normalize(W, dim=1); cos = (Wn @ Wn.T).cpu().numpy(); np.fill_diagonal(cos, np.nan)
            sv = torch.linalg.svdvals(W.float()).cpu().numpy(); pr_ = (sv.sum() ** 2) / (sv ** 2).sum()
            results["heads"][hname] = {"per_phrase": per, "xfer": xfer, "row_cos_mean": float(np.nanmean(cos)), "row_cos_max": float(np.nanmax(cos)),
                                       "participation_ratio": float(pr_), "row_norm_mean": float(W.norm(dim=1).mean())}
            a = [v for v in per.values()]
            print(f"  {hname}: AUROC gen {np.mean([v['vs_generic']['auroc'] for v in a]):.3f}  TPR@1e-3 {np.mean([v['vs_generic']['tpr@fpr0.001'] for v in a]):.3f}  "
                  f"TPR@1e-2 {np.mean([v['vs_generic']['tpr@fpr0.01'] for v in a]):.3f}  sib AUROC {np.mean([v['vs_sibling']['auroc'] for v in a if 'vs_sibling' in v] or [float('nan')]):.3f}  "
                  f"sib TPR@1e-2 {np.mean([v['vs_sibling']['tpr@fpr0.01'] for v in a if 'vs_sibling' in v] or [float('nan')]):.3f}  "
                  f"R-lens earliest L50 median {np.median([v['lens']['rlens']['earliest_layer_tpr50'] for v in a]):.0f}", flush=True)
    os.makedirs(f"{V}/results", exist_ok=True)
    tag = axis if len(axes) == 1 else "joint_" + "_".join(axes)
    json.dump(results, open(f"{V}/results/gen_{tag}.json", "w"), indent=1, ensure_ascii=False)
    vol.commit()
    return results


@app.local_entrypoint()
def main(stage: str = "all", axis: str = "nouns", joint_axes: str = ""):
    out_dir = os.path.join(here, "results"); os.makedirs(out_dir, exist_ok=True)
    langs = sorted({it[3] for it in axis_phrases(axis)})
    if stage in ("mine", "all"):
        for lang in langs:
            for s in mine.map(range(15), kwargs={"axis": axis, "lang": lang}):
                print(lang, s["docs"], min(s["counts"].values()) if s["counts"] else 0)
    if stage in ("merge", "all"):
        merge.remote(axis)
    if stage in ("collect", "all"):
        collect.remote(axis)
    if stage in ("eval", "all"):
        res = evaluate.remote(axis, joint_axes=joint_axes)
        tag = axis if not joint_axes else "joint_" + "_".join(joint_axes.split(","))
        json.dump(res, open(os.path.join(out_dir, f"gen_{tag}.json"), "w"), indent=1, ensure_ascii=False)
        print("wrote", os.path.join(out_dir, f"gen_{tag}.json"))
