"""
phrase_context_miner.py

Scans a large text corpus (streamed, no full download required) for
occurrences of target words/phrases and saves the surrounding context
(a window of sentences around each match) to a JSONL file.

Supported corpora (via HuggingFace `datasets`, streaming=True):
  - "fineweb"    -> HuggingFaceFW/fineweb (Common Crawl, cleaned, English)
  - "pile"       -> monology/pile-uncopyrighted (mirror of The Pile)
  - "wikipedia"  -> wikimedia/wikipedia (20231101.en) -- prose, not Wikidata
  (Wikidata itself is a structured knowledge graph, not prose text, so it's
   not included here -- it won't give you sentence-level "context".)

Usage (CLI flags):
    python phrase_context_miner.py \
        --phrases "blackmail" "New York" \
        --dataset fineweb \
        --samples-per-phrase 150 \
        --context-sentences 2 \
        --out results.jsonl \
        --max-docs 2000000

Usage (YAML config):
    python phrase_context_miner.py --config config.yaml

    # config.yaml:
    phrases:
      - blackmail
      - New York
    dataset: fineweb
    samples_per_phrase: 150
    context_sentences: 2
    out: results.jsonl
    max_docs: 2000000
    case_sensitive: false
    progress_every: 2000

    Any CLI flag passed alongside --config overrides the corresponding
    value from the YAML file (e.g. `--config config.yaml --samples-per-phrase 50`
    runs with everything from the file except samples_per_phrase).

Output: one JSON object per line:
    {"phrase": "...", "context": "...", "doc_id": "...", "source": "..."}
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict

from datasets import load_dataset

# Simple sentence splitter (avoids adding nltk as a hard dependency).
# Splits on '.', '!', '?' followed by whitespace + capital letter/EOF.
# Good enough for corpus mining; not linguistically perfect.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


def split_sentences(text: str):
    text = text.replace("\n", " ")
    # Collapse excess whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENT_SPLIT_RE.split(text)


def build_phrase_regex(phrases, case_sensitive=False):
    """Build one combined regex with word boundaries around each phrase."""
    escaped = [re.escape(p) for p in phrases]
    # Sort longest-first so overlapping phrases don't get shadowed
    escaped.sort(key=len, reverse=True)
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)


def find_context_for_matches(sentences, phrase_regex, context_sentences):
    """
    For each sentence containing a match, yield (matched_phrase, context_str).
    context_str = up to `context_sentences` sentences before the match sentence,
    plus the match sentence itself.
    """
    for i, sent in enumerate(sentences):
        m = phrase_regex.search(sent)
        if not m:
            continue
        start = max(0, i - context_sentences)
        context = " ".join(sentences[start:i + 1]).strip()
        yield m.group(0), context


def load_stream(dataset_name):

    if dataset_name == "fineweb":
        # ~25M docs in the 10BT sample split; streaming avoids the ~30GB download
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )
        text_key = "text"
        id_key = "id"
    elif dataset_name == "pile":
        ds = load_dataset(
            "monology/pile-uncopyrighted", split="train", streaming=True
        )
        text_key = "text"
        id_key = None
    elif dataset_name == "wikipedia":
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
        )
        text_key = "text"
        id_key = "id"
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return ds, text_key, id_key


DEFAULTS = {
    "phrases": None,
    "dataset": "fineweb",
    "samples_per_phrase": 150,
    "context_sentences": 2,
    "out": "results.jsonl",
    "max_docs": 2_000_000,
    "case_sensitive": False,
    "progress_every": 2000,
}


def load_yaml_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Normalize hyphenated keys to underscores, in case a user writes
    # 'samples-per-phrase' instead of 'samples_per_phrase' in the YAML.
    return {k.replace("-", "_"): v for k, v in data.items()}


def resolve_args():
    """
    Two-pass parsing so a --config file can supply defaults while
    explicit CLI flags still take precedence over it.
    """
    # Pass 1: just look for --config, without requiring other args
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                      help="Path to a YAML file supplying any of the options below")
    pre_args, remaining_argv = pre.parse_known_args()

    config_values = {}
    if pre_args.config:
        config_values = load_yaml_config(pre_args.config)

    # Merge: hardcoded defaults < YAML config values
    merged_defaults = {**DEFAULTS, **config_values}

    # Pass 2: full parser. `phrases` becomes optional here since it may
    # come from the YAML file instead of the command line.
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--phrases", nargs="+", default=merged_defaults["phrases"],
                     help="Target words/phrases to search for, e.g. blackmail \"New York\"")
    ap.add_argument("--dataset", choices=["fineweb", "pile", "wikipedia"],
                     default=merged_defaults["dataset"])
    ap.add_argument("--samples-per-phrase", type=int, dest="samples_per_phrase",
                     default=merged_defaults["samples_per_phrase"],
                     help="Stop collecting a phrase once this many samples are found")
    ap.add_argument("--context-sentences", type=int, dest="context_sentences",
                     default=merged_defaults["context_sentences"],
                     help="Number of sentences before the matching sentence to include")
    ap.add_argument("--out", default=merged_defaults["out"])
    ap.add_argument("--max-docs", type=int, dest="max_docs",
                     default=merged_defaults["max_docs"],
                     help="Safety cap on number of documents scanned before giving up")
    ap.add_argument("--case-sensitive", dest="case_sensitive", action="store_true",
                     default=merged_defaults["case_sensitive"])
    ap.add_argument("--progress-every", type=int, dest="progress_every",
                     default=merged_defaults["progress_every"],
                     help="Print a progress line every N documents scanned")
    args = ap.parse_args()

    if not args.phrases:
        ap.error("No phrases given. Provide --phrases on the CLI or a 'phrases' "
                  "list in the --config YAML file.")

    return args


def main():
    args = resolve_args()

    phrase_regex = build_phrase_regex(args.phrases, args.case_sensitive)
    target = args.samples_per_phrase
    counts = defaultdict(int)

    print(f"Loading '{args.dataset}' in streaming mode...", file=sys.stderr)
    ds, text_key, id_key = load_stream(args.dataset)

    t0 = time.time()
    docs_scanned = 0
    docs_with_hit = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for doc in ds:
            docs_scanned += 1
            text = doc.get(text_key, "")
            if not text:
                continue

            # Fast pre-filter: skip sentence splitting if no phrase appears at all
            if not phrase_regex.search(text):
                pass
            else:
                docs_with_hit += 1
                sentences = split_sentences(text)
                for matched_phrase, context in find_context_for_matches(
                    sentences, phrase_regex, args.context_sentences
                ):
                    key = matched_phrase.lower() if not args.case_sensitive else matched_phrase
                    if counts[key] >= target:
                        continue
                    counts[key] += 1
                    record = {
                        "phrase": matched_phrase,
                        "context": context,
                        "doc_id": doc.get(id_key) if id_key else docs_scanned,
                        "source": args.dataset,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if docs_scanned % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = docs_scanned / elapsed if elapsed > 0 else 0
                done_str = ", ".join(f"{p}={counts.get(p.lower(), 0)}/{target}" for p in args.phrases)
                print(
                    f"[{elapsed:6.1f}s] scanned={docs_scanned:,} "
                    f"({rate:.0f} docs/s) hits_docs={docs_with_hit:,} | {done_str}",
                    file=sys.stderr,
                )

            # Stop once every requested phrase has hit its target
            if all(counts.get(p.lower() if not args.case_sensitive else p, 0) >= target
                   for p in args.phrases):
                print("All phrase quotas reached.", file=sys.stderr)
                break

            if docs_scanned >= args.max_docs:
                print("Reached --max-docs safety cap before all quotas were filled.",
                      file=sys.stderr)
                break

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Scanned {docs_scanned:,} documents.", file=sys.stderr)
    for p in args.phrases:
        k = p.lower() if not args.case_sensitive else p
        print(f"  '{p}': {counts.get(k, 0)} samples collected", file=sys.stderr)
    print(f"Output written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()