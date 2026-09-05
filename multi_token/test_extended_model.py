"""
Tests to ensure that extende model behaves as intended.
Correct behavior is vital for understanding experimental outputs.

Currently embed method must be set manually.

"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import random

import torch

from extend_model import generate_extended_tok_and_model, EmbedMethod

MODEL_NAME = "Qwen/Qwen3.5-9B"

# Create base model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Modify model
model, extended_tokenizer = generate_extended_tok_and_model(
    model=model, 
    tokenizer=tokenizer, 
    emb_method=EmbedMethod.PRIOR_REPRESENTATION_EMBED_PROJ,
    data_path="./results/phrase_means.pt"
)

def test_encoding_stable():
    input = "blackmail"
    original_tokenization = tokenizer.encode(input)
    new_tokenization = extended_tokenizer.encode(input)

    assert len(original_tokenization) == len(new_tokenization)
    for i in range(len(original_tokenization)):
        assert original_tokenization[i] == new_tokenization[i]

def test_decode_original_vocab_stable():
    random_id = random.randint(0, len(tokenizer) - 1)

    original_decode = tokenizer.decode([random_id])
    new_decode = extended_tokenizer.decode([random_id])

    assert original_decode == new_decode

def test_decode_new_token():
    new_id = len(tokenizer)

    assert extended_tokenizer.decode([new_id]) == "New Hampshire"