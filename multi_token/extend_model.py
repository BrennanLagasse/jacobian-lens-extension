"""
To extend the j-lens to multi-token phrases, we consider a transformer
with an extended output vocabulary that includes select multi-token
phrases as individual tokens. Given a set of results from approximating
the representation of these phrases, this file supports updating the 
unembedding matrix of the model and wrapping the tokenizer so it is
able to decode in this space with no change to encoding 

The output is intended for use for jlens analysis, not generative
tasks
"""

import torch
import torch.nn.functional as F

from enum import Enum

class EmbedMethod(Enum):
    AVERAGE_TOKEN_WEIGHTS = 1
    PRIOR_REPRESENTATION_EMBED = 2

def generate_extended_tok_and_model(model, tokenizer, emb_method, data_path):
    """ Generate the extended tokenizer and model at the same time
    to ensure that the pair align 
    
    Arguments:
        model (AutoModelForCausalLM): base model
        tokenizer: base tokenizer
        emb_method: way to generate weights for unembedding phrase
        data_path: path to file containing weights 

    Return:
        new_model: model augmented with new head weights to decode new phrases
        new_tokenizer: wrapped tokenizer that can decode new phrases
    """

    assert isinstance(emb_method, EmbedMethod)

    if emb_method == EmbedMethod.AVERAGE_TOKEN_WEIGHTS:

        # TODO: clean this up later

        data = torch.load(data_path, map_location="cpu")
        
        new_phrases = list(data['phrase_means'].keys()) 

        # Compute mean weights of the phrase
        encoded_phrases = tokenizer.encode(new_phrases)
        lm_head_weights = model.get_output_embeddings().weight
        indices = torch.tensor([idx for phrase in encoded_phrases for idx in phrase], device=lm_head_weights.device)
        weights = lm_head_weights.index_select(0, indices)
        lengths = [len(phrase) for phrase in encoded_phrases]
        split_weights = torch.split(weights, lengths)
        new_phrase_weights = torch.stack([phrase_weights.mean(dim=0) for phrase_weights in split_weights])

    if emb_method == EmbedMethod.PRIOR_REPRESENTATION_EMBED:

        data = torch.load(data_path, map_location="cpu")

        new_phrases = list(data['phrase_means'].keys())

        # Stabilize weights by subtracting baseline and normalizing in L2

        h_mean = torch.load("./results/h_bar.pt")

        new_phrase_weights = torch.stack(
            [F.normalize(h - h_mean, p=2, dim=0) for h in data['phrase_means'].values()], 
            dim=0
        )

    updated_model = extend_model(model, tokenizer, new_phrase_weights)
    updated_tokenizer = DecodeExtendedTokenizer(tokenizer, new_phrases)
        
    return updated_model, updated_tokenizer


import torch

def extend_model(model, tokenizer, new_phrase_weights):
    """
    Extend the decoder to emit n new tokens (ids len(tokenizer):len(tokenizer)+n).
    Input embeddings are also grown to keep dimensions consistent with the
    rest of the HF framework, but the new input rows are zero-initialized
    and frozen (never updated, never intended to be used) since these
    tokens are never fed back into the encoder.
    """

    old_len = len(tokenizer)
    n = new_phrase_weights.shape[0]
    target_len = old_len + n

    if model.config.tie_word_embeddings:
        model.config.tie_word_embeddings = False
        model.get_output_embeddings().weight = torch.nn.Parameter(
            model.get_output_embeddings().weight.data.clone()
        )

    # Grow both embeddings via HF's own logic, so shapes/attrs stay consistent
    model.resize_token_embeddings(target_len)

    lm_head = model.get_output_embeddings()
    input_emb = model.get_input_embeddings()

    new_phrase_weights = new_phrase_weights.to(
        device=lm_head.weight.device, dtype=lm_head.weight.dtype
    )
    with torch.no_grad():
        lm_head.weight[old_len:target_len] = new_phrase_weights
        input_emb.weight[old_len:target_len] = 0.0  # unused; never fed as input

    # Freeze just the new input rows so they can't drift during training,
    # since a plain requires_grad=False would freeze the WHOLE embedding table.
    def _zero_new_input_rows_grad(grad):
        grad = grad.clone()
        grad[old_len:target_len] = 0
        return grad

    input_emb.weight.register_hook(_zero_new_input_rows_grad)

    model.config.vocab_size = target_len
    return model

class DecodeExtendedTokenizer:
    """ Wrap a tokenizer to decode on an extended vocab (and change nothing else) """

    def __init__(self, base_tokenizer, new_token_strs):
        self._tok = base_tokenizer
        self._extra_lookup = {len(base_tokenizer) + i: s for i, s in enumerate(new_token_strs)}

    def decode(self, ids, **kwargs):
        ids = list(ids)
        # fast path: jlens calls tok.decode([t]) with a single id at a time
        if len(ids) == 1 and int(ids[0]) in self._extra_lookup:
            return self._extra_lookup[int(ids[0])]
            
        # general path: reconstruct in order, in case of mixed id lists
        pieces, buf = [], []
        for i in ids:
            if i in self._extra_lookup:
                if buf:
                    pieces.append(self._tok.decode(buf, **kwargs)); buf = []
                pieces.append(self._extra_lookup[i])
            else:
                buf.append(i)
        if buf:
            pieces.append(self._tok.decode(buf, **kwargs))
        return "".join(pieces)

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tok, name) 

    def __len__(self):
        return len(self._tok) + len(self._extra_lookup)