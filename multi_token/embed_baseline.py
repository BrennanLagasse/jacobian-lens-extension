"""
Approximate h_bar: the mean hidden state fed into lm_head (i.e. post-final-norm,
pre-unembedding), averaged over "generic" text.

Used to de-bias newly-added unembedding rows: subtract h_bar from a token's
mean activation before normalizing, so the new row doesn't inherit the
implicit high-frequency / attention-sink bias that all hidden states share.

Usage:
    python compute_h_bar.py \
        --model Qwen/Qwen3.5-9B \
        --n-docs 2000 \
        --tokens-per-doc 512 \
        --out h_bar.pt
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def get_final_norm_module(model):
    """
    Locate the model's final normalization layer -- the one applied just
    before lm_head. We hook this module's output directly rather than
    hidden_states[-1] from output_hidden_states=True, because the latter
    is ambiguous across architectures about whether the final norm has
    already been applied.
    """
    # Qwen3.5 (and most Qwen/Llama-family models): model.model.norm
    base = getattr(model, model.base_model_prefix, model)
    if hasattr(base, "norm"):
        return base.norm
    raise AttributeError(
        "Could not locate final norm module automatically -- "
        "inspect `model` and set it manually, e.g. model.model.norm"
    )


@torch.no_grad()
def compute_h_bar(
    model_name: str,
    n_docs: int,
    tokens_per_doc: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    skip_bos_tokens: int,
):
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model.eval()

    norm_module = get_final_norm_module(model)

    # Accumulate sum and count of post-norm hidden states, streamed via a
    # forward hook so we never have to hold all activations in memory.
    running_sum = torch.zeros(model.config.hidden_size, dtype=torch.float64, device=device)
    running_count = torch.zeros(1, dtype=torch.float64, device=device)

    def hook(_module, _inp, output):
        # output: (batch, seq_len, hidden_size)
        hs = output.detach().to(torch.float64)
        # Drop the first `skip_bos_tokens` positions per sequence: early
        # positions (esp. position 0 / BOS) are well-documented attention
        # sinks with anomalously large-magnitude hidden states and would
        # skew the mean if included.
        hs = hs[:, skip_bos_tokens:, :]
        running_sum.add_(hs.reshape(-1, hs.shape[-1]).sum(dim=0))
        running_count.add_(hs.shape[0] * hs.shape[1])

    handle = norm_module.register_forward_hook(hook)

    print("Streaming FineWeb...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    docs_processed = 0
    batch_texts = []

    def flush_batch(texts):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokens_per_doc,
        ).to(device)
        model(**enc, use_cache=False)

    try:
        for example in tqdm(ds):
            if docs_processed >= n_docs:
                break
            text = example.get("text", "")
            if not text.strip():
                continue
            batch_texts.append(text)
            docs_processed += 1

            if len(batch_texts) == batch_size:
                flush_batch(batch_texts)
                batch_texts = []
                if docs_processed % (batch_size * 10) == 0:
                    print(f"  {docs_processed}/{n_docs} docs processed")

        if batch_texts:
            flush_batch(batch_texts)
    finally:
        handle.remove()

    h_bar = (running_sum / running_count).to(torch.float32).cpu()
    print(f"Done. Averaged over {int(running_count.item())} token positions "
          f"across {docs_processed} documents.")
    print(f"||h_bar|| = {h_bar.norm().item():.4f}")
    return h_bar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--n-docs", type=int, default=2000,
                         help="Number of FineWeb documents to average over")
    parser.add_argument("--tokens-per-doc", type=int, default=512,
                         help="Max tokens per document (also used as truncation length)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16",
                         choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-bos-tokens", type=int, default=4,
                         help="Number of leading positions per sequence to exclude "
                              "(attention-sink / massive-activation positions)")
    parser.add_argument("--out", default="h_bar.pt")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    h_bar = compute_h_bar(
        model_name=args.model,
        n_docs=args.n_docs,
        tokens_per_doc=args.tokens_per_doc,
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        skip_bos_tokens=args.skip_bos_tokens,
    )

    torch.save(h_bar, args.out)
    print(f"Saved h_bar to {args.out}")


if __name__ == "__main__":
    main()