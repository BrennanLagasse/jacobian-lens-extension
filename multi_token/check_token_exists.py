from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-9B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def inspect(word: str):
    print("=" * 70)
    print(f"Input: {word!r}")

    # 1. Normal tokenization
    ids = tokenizer.encode(word, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("\nEncoding:")
    print(f"  IDs:    {ids}")
    print(f"  Tokens: {tokens}")
    print(f"  Count:  {len(ids)}")
    print(f"  Decode: {tokenizer.decode(ids)!r}")

    if len(ids) == 1:
        print(f"\n✅ Tokenizes to ONE token: {ids[0]}")
    else:
        print(f"\n❌ Tokenizes to {len(ids)} tokens")

    # 2. Check whether the exact string appears in the vocab
    vocab = tokenizer.get_vocab()

    if word in vocab:
        token_id = vocab[word]
        print(f"\n📖 Exact vocabulary entry found:")
        print(f"  String: {word!r}")
        print(f"  ID:     {token_id}")
    else:
        print("\n📖 Exact string is NOT a vocabulary key.")

    # 3. Look for vocabulary entries that decode to the same string
    matching = []

    for token, token_id in vocab.items():
        try:
            decoded = tokenizer.decode([token_id])
            if decoded == word:
                matching.append((token, token_id))
        except Exception:
            pass

    if matching:
        print("\n🔎 Vocabulary tokens that decode to the input:")
        for token, token_id in matching:
            print(f"  {token!r} -> {token_id}")
    else:
        print("\n🔎 No single vocabulary token decodes exactly to this string.")


if __name__ == "__main__":

    word = "blackmail"

    inspect(word)
