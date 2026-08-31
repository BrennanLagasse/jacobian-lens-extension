#!/usr/bin/env python3
"""
extract_representations.py

Given the JSONL output of phrase_context_miner.py (records with "phrase"
and "context"), compute the model's final latent representation (the last
transformer hidden state, BEFORE the lm_head/vocab projection) at the
token position associated with the LAST occurrence of the target phrase
in each context. Runs efficiently in batches on GPU.

-----------------------------------------------------------------------
WHICH TOKEN POSITION, EXACTLY?
-----------------------------------------------------------------------
"the representation preceding the final occurrence of the target word"
is ambiguous, so this script supports three modes via --position:

  before  (default) -> hidden state at the LAST token immediately BEFORE
                        the phrase starts. This is "the context the model
                        had accumulated right before it saw the phrase" --
                        i.e. the representation that, if you multiplied it
                        by the LM head, would predict the phrase's first
                        token. This matches the literal wording "preceding
                        the occurrence".
  last_token          -> hidden state at the LAST token OF the phrase
                        itself (context including the phrase).
  after               -> hidden state at the first token AFTER the phrase
                        ends (context including the phrase plus one more
                        token of "look-ahead" from the model's causal
                        attention).

Pick whichever matches your downstream use. If in doubt, `before` is the
standard choice for "what did the model know right before this word".

-----------------------------------------------------------------------
EFFICIENCY DESIGN
-----------------------------------------------------------------------
- Skip the LM head entirely. We call the base transformer submodule
  (`model.model`, standard for Qwen/Llama-style causal LMs) directly,
  rather than the full CausalLM wrapper. This avoids ever materializing
  the (batch, seq_len, vocab_size) logits tensor -- for a ~150k
  vocabulary that tensor is large and completely wasted compute/memory
  here since we only want hidden states.
- use_cache=False. We're doing a single forward pass per sequence, not
  autoregressive generation, so there's no reason to allocate a KV cache.
- torch.no_grad() + eval() + bf16. No gradients needed; bf16 halves
  weight/activation memory vs fp32 with negligible accuracy impact for
  this kind of feature extraction.
- Length-sorted bucketing. Contexts are short (~1-2 sentences) but still
  vary in length. Sorting by token length before batching means each
  batch pads to the length of its longest *local* member instead of the
  longest member in the whole dataset, cutting wasted compute on pad
  tokens substantially.
- Right-padding with an explicit attention_mask, so target-token indices
  computed from the unpadded tokenization stay valid without offset
  arithmetic (left-padding would require adding the pad offset to every
  index).

-----------------------------------------------------------------------
Usage:
    python collect_embeddings.py \
        --input results.jsonl \
        --model Qwen/Qwen3.5-9B \
        --position before \
        --out embeddings.pt \
        --batch-size auto
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
import re
import sys
import time
import math

import torch

from tqdm import tqdm


def load_model_and_tokenizer(model_name, dtype=torch.bfloat16, device="cuda"):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Common for causal-LM checkpoints; padding needs *a* token id,
        # it's masked out by attention_mask so the value itself doesn't
        # matter for correctness.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        # attn_implementation="sdpa",  # falls back gracefully if flash-attn isn't installed
    ).to(device)
    full_model.eval()

    # `.model` is the bare transformer (embeddings -> decoder layers -> final
    # norm) without the lm_head projection, for standard HF causal-LM
    # architectures (Qwen*, Llama, Mistral, etc). This IS the "final latent
    # representation prior to the projection head" the task asks for.
    base_model = full_model.model
    return base_model, tokenizer


def find_target_span(context, phrase, case_sensitive=False):
    """Return (char_start, char_end) of the LAST occurrence of phrase in context, or None."""
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", flags)
    matches = list(pattern.finditer(context))
    if not matches:
        return None
    m = matches[-1]
    return m.start(), m.end()


def char_span_to_token_index(offset_mapping, char_start, char_end, position):
    """
    offset_mapping: list of (start, end) char offsets, one per token
                     (from a fast tokenizer's `return_offsets_mapping=True`).
    Returns a single token index according to `position` mode.
    """
    token_start = token_end = None
    for i, (s, e) in enumerate(offset_mapping):
        if s == e:
            continue  # special tokens have (0, 0) offsets
        if s < char_end and e > char_start:  # overlaps the phrase span
            if token_start is None:
                token_start = i
            token_end = i

    if token_start is None:
        return None

    if position == "before":
        return max(0, token_start - 1)
    elif position == "last_token":
        return token_end
    elif position == "after":
        return min(len(offset_mapping) - 1, token_end + 1)
    else:
        raise ValueError(f"Unknown position mode: {position}")


def prepare_examples(records, tokenizer, position, case_sensitive):
    """
    Tokenize every record once (CPU-side, cheap) and resolve the target
    token index. Records where the phrase isn't found, or where the
    resolved index would need context the tokenizer doesn't have, are
    skipped with a warning.
    """
    examples = []
    skipped = 0
    for rec in records:
        context, phrase = rec["context"], rec["phrase"]
        span = find_target_span(context, phrase, case_sensitive)
        if span is None:
            skipped += 1
            continue

        enc = tokenizer(context, return_offsets_mapping=True, add_special_tokens=True)
        idx = char_span_to_token_index(enc["offset_mapping"], span[0], span[1], position)
        if idx is None:
            skipped += 1
            continue

        examples.append({
            "input_ids": enc["input_ids"],
            "target_index": idx,
            "record": rec,
        })

    if skipped:
        print(f"Skipped {skipped}/{len(records)} records (phrase/offset resolution failed).",
              file=sys.stderr)

    # Sort by length for bucketed batching -- see module docstring.
    examples.sort(key=lambda ex: len(ex["input_ids"]))
    return examples


def make_batches(examples, batch_size):
    for i in range(0, len(examples), batch_size):
        yield examples[i:i + batch_size]


@torch.no_grad()
def run_batch(base_model, tokenizer, batch, device):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = ex["input_ids"]
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1

    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    out = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state  # (B, L, H) -- pre-projection-head representations

    idx = torch.tensor([ex["target_index"] for ex in batch], device=device)
    selected = hidden[torch.arange(len(batch), device=device), idx]  # (B, H)
    return selected.to("cpu", torch.float32)


# -----------------------------------------------------------------------
# Batch-size calibration
# -----------------------------------------------------------------------

@torch.no_grad()
def find_max_batch_size(base_model, tokenizer, device, probe_seq_len, max_probe_batch=1024):
    """
    Empirically determine the largest batch size that fits in GPU memory
    for sequences of length `probe_seq_len` (use the LONGEST sequence
    length in your actual dataset, not the average -- that's your worst
    case and the one that determines whether you OOM mid-run).

    Strategy: exponential search (1, 2, 4, 8, ...) to find an upper
    bound that OOMs, then binary search between the last success and
    first failure. Returns the largest batch size that succeeded.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def try_batch(bs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            input_ids = torch.full((bs, probe_seq_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.ones((bs, probe_seq_len), dtype=torch.long, device=device)
            base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            return True, peak_gb
        except torch.cuda.OutOfMemoryError:
            return False, None
        finally:
            torch.cuda.empty_cache()

    last_good, last_good_mem = 0, 0.0
    bs = 1
    while bs <= max_probe_batch:
        ok, mem = try_batch(bs)
        print(f"  probe batch_size={bs:<5} -> {'OK' if ok else 'OOM'}"
              + (f" (peak {mem:.2f} GB)" if ok else ""), file=sys.stderr)
        if not ok:
            break
        last_good, last_good_mem = bs, mem
        bs *= 2
    else:
        print(f"Reached max_probe_batch={max_probe_batch} without OOM; "
              f"you likely don't need to search further.", file=sys.stderr)
        return last_good

    # Binary search between last_good and bs (which OOM'd)
    lo, hi = last_good, bs
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, mem = try_batch(mid)
        print(f"  probe batch_size={mid:<5} -> {'OK' if ok else 'OOM'}"
              + (f" (peak {mem:.2f} GB)" if ok else ""), file=sys.stderr)
        if ok:
            lo, last_good, last_good_mem = mid, mid, mem
        else:
            hi = mid

    print(f"Max feasible batch size at seq_len={probe_seq_len}: {last_good} "
          f"(peak memory ~{last_good_mem:.2f} GB)", file=sys.stderr)
    return last_good


# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL from phrase_context_miner.py")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--position", choices=["before", "last_token", "after"], default="before")
    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--out", default="embeddings.pt")
    ap.add_argument("--batch-size", default="auto",
                     help="Integer, or 'auto' to calibrate against GPU memory")
    ap.add_argument("--safety-margin", type=float, default=0.85,
                     help="Fraction of the empirically max-feasible batch size to actually use "
                          "(headroom for fragmentation / other processes / longer-than-probed inputs)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("\nSetting up...")
    records = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    print(f"Loaded {len(records)} records from {args.input}", file=sys.stderr)

    base_model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)

    examples = prepare_examples(records, tokenizer, args.position, args.case_sensitive)
    if not examples:
        print("No usable examples after tokenization; nothing to do.", file=sys.stderr)
        return

    if args.batch_size == "auto":
        probe_len = len(examples[-1]["input_ids"])  # longest sequence in the dataset
        print(f"Calibrating batch size against longest sequence "
              f"({probe_len} tokens)...", file=sys.stderr)
        max_bs = find_max_batch_size(base_model, tokenizer, args.device, probe_len)
        batch_size = max(1, int(max_bs * args.safety_margin))
        print(f"Using batch size {batch_size} "
              f"({args.safety_margin:.0%} of empirical max {max_bs}).", file=sys.stderr)
    else:
        batch_size = int(args.batch_size)

    print("\nComputing all embeddings...")
    print(f"\nStarting {math.ceil(len(records) / batch_size)} batches of size {batch_size}\n")
    all_embeddings, all_records = [], []
    t0 = time.time()
    n_done = 0
    for batch in tqdm(make_batches(examples, batch_size)):
        emb = run_batch(base_model, tokenizer, batch, args.device)
        all_embeddings.append(emb)
        all_records.extend(ex["record"] for ex in batch)
        n_done += len(batch)
        if n_done % (batch_size * 10) < batch_size:
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s] {n_done}/{len(examples)} "
                  f"({n_done / elapsed:.1f} samples/s)", file=sys.stderr)

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, hidden_dim)
    torch.save({"embeddings": embeddings, "records": all_records}, args.out)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved {embeddings.shape[0]} embeddings "
          f"of dim {embeddings.shape[1]} to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()