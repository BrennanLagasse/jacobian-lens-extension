#!/usr/bin/env python3
"""
average_phrase_representations.py

Takes the .pt file produced by extract_representations.py (a dict with
"embeddings" of shape (N, d_model) and "records" of length N, each with
a "phrase" field) and averages the vectors belonging to each unique
phrase, producing one d_model-dimensional vector per phrase.

Usage:
    python process_representations.py \
        --input results/embeddings.pt \
        --out results/phrase_means.pt \
        --case-insensitive
"""

import argparse
from collections import defaultdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".pt file from extract_representations.py")
    ap.add_argument("--out", default="phrase_means.pt")
    ap.add_argument("--case-insensitive", action="store_true",
                     help="Merge phrase variants that differ only in case (e.g. 'New York' "
                          "and 'new york') into a single average. Off by default so the "
                          "output keys match the exact 'phrase' strings recorded upstream.")
    args = ap.parse_args()

    data = torch.load(args.input, map_location="cpu")
    embeddings = data["embeddings"]  # (N, d_model)
    records = data["records"]        # length-N list of dicts with a "phrase" field

    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"Mismatch: {embeddings.shape[0]} embeddings but {len(records)} records."
        )

    # Group embedding row indices by phrase key
    groups = defaultdict(list)
    display_name = {}  # key -> a representative original-cased phrase string, for readability
    for i, rec in enumerate(records):
        phrase = rec["phrase"]
        key = phrase.lower() if args.case_insensitive else phrase
        groups[key].append(i)
        display_name.setdefault(key, phrase)

    phrase_means = {}
    counts = {}
    for key, idxs in groups.items():
        vecs = embeddings[idxs]              # (n_i, d_model)
        mean_vec = vecs.mean(dim=0)          # (d_model,)
        name = display_name[key]
        phrase_means[name] = mean_vec
        counts[name] = len(idxs)

    torch.save({"phrase_means": phrase_means, "counts": counts}, args.out)

    print(f"Averaged {embeddings.shape[0]} vectors (dim={embeddings.shape[1]}) "
          f"into {len(phrase_means)} unique phrase(s):")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  '{name}': {n} samples averaged")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()