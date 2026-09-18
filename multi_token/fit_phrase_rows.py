#!/usr/bin/env python3
"""
fit_phrase_rows.py

Fit unembedding rows for multi-token phrases by logistic regression over the
*extended* softmax, so the new rows live in the same logit units as W_U.

Why not a prototype direction (PRIOR_REPRESENTATION_EMBED*): a centred,
unit-norm mean state has the right norm but not the right variance against the
residual stream. On Qwen3.5-9B-Base its logit std over generic text is ~18 vs
~1.8 for a real W_U row, so it sits in the top-10 at ~19% of generic positions
and dominates every lens readout regardless of context. The rows fitted here
put mass on the new tokens equal to the corpus prior, have a generic top-10
rate near 0, AUROC ~0.98 (own pre-phrase positions vs generic), and rank the
phrase within ~1.25x of the rank of its real first token.

Recipe (no bias, like a real W_U row):
  positives  post-final-norm state at the token before each mined occurrence
             (the state lens.apply unembeds); target = the phrase token
  negatives  every position of N generic FineWeb docs; target = real next token
  loss       cross-entropy over softmax([real logits, new logits]); positives
             weighted by pi_corpus / pi_train instead of fitting a bias; L2 on W
  optimiser  full-batch L-BFGS (chunked, so it runs on any device / CPU).
             Minibatch Adam under-fits this; a bias term makes the rows fire at
             early lens layers.

Usage:
    python fit_phrase_rows.py --contexts results.jsonl \
        --model Qwen/Qwen3.5-9B-Base --n-generic-docs 2000 --out results/phrase_rows.pt
    # then: EmbedMethod.FITTED_ROWS, data_path="results/phrase_rows.pt"

Memory: generic states are kept on CPU in bf16, n_docs * tokens_per_doc * d_model * 2 bytes
(2000 x 512 x 4096 -> ~8 GB). Lower --n-generic-docs on small machines.
"""

import argparse
import itertools
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_embeddings import char_span_to_token_index, find_target_span

SKIP = 4     # leading positions excluded (attention sink), as in embed_baseline.py
TOPK = 100   # real logits kept per held-out positive, for the rank report


@torch.no_grad()
def states_and_logits(model, ids, device):
    """Post-final-norm states [L, d] and logits [L, V] for one unpadded sequence."""
    h = model.model(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state[0]
    return h, model.get_output_embeddings()(h).float()


@torch.no_grad()
def collect_positives(model, tok, records, pidx, device):
    out, skipped = [], 0
    for rec in records:
        span = find_target_span(rec["context"], rec["phrase"])
        enc = tok(rec["context"], return_offsets_mapping=True)
        i = None if span is None else char_span_to_token_index(enc["offset_mapping"], *span, "before")
        if i is None:
            skipped += 1
            continue
        h, z = states_and_logits(model, enc["input_ids"], device)
        z = z[i]
        out.append({
            "h": h[i].to("cpu", torch.bfloat16),
            "lse": torch.logsumexp(z, 0).item(),
            "top": z.topk(TOPK).values.cpu(),
            "first_rank": int((z > z[enc["input_ids"][i + 1]]).sum()) + 1,
            "phrase": pidx[rec["phrase"].lower()],
        })
    print(f"positives: {len(out)} usable, {skipped} skipped", file=sys.stderr)
    return {k: torch.stack([o[k] for o in out]) if k in ("h", "top") else torch.tensor([o[k] for o in out]) for k in out[0]}


@torch.no_grad()
def collect_generic(model, tok, n_docs, max_len, device, seed):
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    H, LSE, ZT, TOP10, DOC, n_chars, n_tok = [], [], [], [], [], 0, 0
    for d, doc in enumerate(itertools.islice(ds.shuffle(seed=seed, buffer_size=10_000), n_docs)):
        ids = tok(doc["text"])["input_ids"]
        n_chars += len(doc["text"]); n_tok += len(ids)
        ids = ids[:max_len]
        if len(ids) < SKIP + 2:
            continue
        h, z = states_and_logits(model, ids, device)
        z, nxt = z[SKIP:-1], torch.tensor(ids[SKIP + 1:], device=device)
        H.append(h[SKIP:-1].to("cpu", torch.bfloat16))
        LSE.append(torch.logsumexp(z, 1).cpu())
        ZT.append(z.gather(1, nxt[:, None]).squeeze(1).cpu())
        TOP10.append(z.topk(10).values[:, -1].cpu())
        DOC.append(torch.full((len(z),), d))
        if d % 100 == 0:
            print(f"generic: {d}/{n_docs} docs", file=sys.stderr)
    return {"h": torch.cat(H), "lse": torch.cat(LSE), "zt": torch.cat(ZT), "top10": torch.cat(TOP10),
            "doc": torch.cat(DOC)}, n_chars / n_tok


def fit_rows(P, G, weights, l2, device, chunk=32768):
    """Bias-free multinomial logistic regression over [real vocab, k new rows]; positives re-weighted."""
    H = torch.cat([P["h"], G["h"]])
    lse = torch.cat([P["lse"], G["lse"]]).float().to(device)
    ph = torch.cat([P["phrase"], torch.full((len(G["h"]),), -1)]).to(device)
    zt = torch.cat([torch.zeros(len(P["h"])), G["zt"]]).float().to(device)
    w = torch.where(ph >= 0, weights.to(device)[ph.clamp(min=0)], torch.ones(len(H), device=device)).double()
    wsum, k = w.sum(), len(weights)
    W = torch.zeros(k, H.shape[1], device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=300, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        opt.zero_grad()
        total = 0.0
        for s in range(0, len(H), chunk):
            sl = slice(s, s + chunk)
            z = H[sl].to(device).float() @ W.T
            L = torch.logsumexp(torch.cat([lse[sl, None], z], 1), 1)
            z_target = torch.where(ph[sl] >= 0, z.gather(1, ph[sl].clamp(min=0)[:, None]).squeeze(1), zt[sl])
            loss = ((L - z_target).double() * w[sl]).sum() / wsum
            loss.backward()
            total += loss.item()
        reg = l2 * W.pow(2).sum()
        reg.backward()
        return total + reg.item()

    opt.step(closure)
    return W.detach().cpu()


@torch.no_grad()
def report(W, phrases, P, G, prior):
    """Held-out calibration: the numbers that decide whether a row behaves like a W_U row."""
    zp, zg = P["h"].float() @ W.T, G["h"].float() @ W.T
    mass = torch.exp(torch.logsumexp(zg, 1) - torch.logsumexp(torch.cat([G["lse"][:, None], zg], 1), 1)).mean()
    print(f"\n{'phrase':<22}{'AUROC':>7}{'gen top10':>11}{'rank own':>10}{'rank 1st tok':>14}{'|row|':>7}")
    for p, name in enumerate(phrases):
        own, gen = zp[P["phrase"] == p, p], zg[:, p]
        auroc = torch.stack([(gen < o).float().mean() for o in own]).mean()
        rank = ((P["top"][P["phrase"] == p] > own[:, None]).sum(1) + 1).clamp(max=TOPK + 1).float().median()
        print(f"{name:<22}{auroc:7.3f}{(gen > G['top10']).float().mean():11.4f}{rank:10.0f}"
              f"{P['first_rank'][P['phrase'] == p].float().median():14.0f}{W[p].norm():7.2f}")
    print(f"mass on new tokens over generic text: {mass:.2e}   corpus prior: {prior.sum():.2e}  "
          f"(ratio {mass / prior.sum():.2f}; ~1 is calibrated, the prototype rows give ~1e4)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True, help="JSONL from phrase_context_miner.py")
    ap.add_argument("--stats", default=None, help="<contexts>.stats.json from the miner (corpus prior)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--n-generic-docs", type=int, default=2000)
    ap.add_argument("--tokens-per-doc", type=int, default=512)
    ap.add_argument("--held-out-frac", type=float, default=0.2, help="per-phrase contexts (and generic docs) kept for the report")
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, help="model dtype; default bfloat16 on GPU, float32 on CPU")
    ap.add_argument("--out", default="results/phrase_rows.pt")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.contexts, encoding="utf-8") if l.strip()]
    phrases = list({r["phrase"].lower(): r["phrase"] for r in records}.values())
    pidx = {p.lower(): i for i, p in enumerate(phrases)}

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float32 if args.device == "cpu" else torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()

    P = collect_positives(model, tok, records, pidx, args.device)
    G, chars_per_tok = collect_generic(model, tok, args.n_generic_docs, args.tokens_per_doc, args.device, args.seed)

    stats_path = args.stats or args.contexts + ".stats.json"
    if os.path.exists(stats_path):
        st = json.load(open(stats_path))
        occ = torch.tensor([st["occ"].get(p.lower(), 0) for p in phrases], dtype=torch.float)
        prior = occ / (st["chars"] / chars_per_tok)
    else:
        raise SystemExit(f"{stats_path} not found: re-run phrase_context_miner.py (it now writes <out>.stats.json) or pass --stats")
    print("corpus prior per token:", {p: f"{x:.2e}" for p, x in zip(phrases, prior.tolist())}, file=sys.stderr)

    # held-out split: per-phrase tail of a seeded permutation; generic by document
    g = torch.Generator().manual_seed(args.seed)
    held = torch.zeros(len(P["h"]), dtype=torch.bool)
    for p in range(len(phrases)):
        idx = (P["phrase"] == p).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        held[idx[: max(1, int(args.held_out_frac * len(idx)))]] = True
    gheld = G["doc"] >= int((1 - args.held_out_frac) * args.n_generic_docs)
    sub = lambda D, m: {k: v[m] for k, v in D.items()}
    Ptr, Gtr = sub(P, ~held), sub(G, ~gheld)

    n_tot = len(Ptr["h"]) + len(Gtr["h"])
    pi_train = torch.bincount(Ptr["phrase"], minlength=len(phrases)).float() / n_tot
    W = fit_rows(Ptr, Gtr, prior / pi_train, args.l2, args.device)

    report(W, phrases, sub(P, held), sub(G, gheld), prior)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"phrase_rows": {p: W[i] for i, p in enumerate(phrases)}, "prior": dict(zip(phrases, prior.tolist())),
                "l2": args.l2, "model": args.model, "n_generic_docs": args.n_generic_docs}, args.out)
    print(f"saved {len(phrases)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
