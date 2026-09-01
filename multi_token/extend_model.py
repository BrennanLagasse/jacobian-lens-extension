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

class embed_method(Enum):
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

    if emb_method == embed_method.AVERAGE_TOKEN_WEIGHTS:
        raise NotImplementedError

    if emb_method == embed_method.PRIOR_REPRESENTATION_EMBED:

        data = torch.load(data_path, map_location="cpu")

        new_phrases = list(data['phrase_means'].keys())

        # Normalize (L2) and stack embeddings
        new_phrase_weights = torch.stack(
            [F.normalize(v, p=2, dim=0) for v in data['phrase_means'].values()], 
            dim=0
        )

        updated_model = extend_model(model, new_phrase_weights)
        updated_tokenizer = DecodeExtendedTokenizer(tokenizer, new_phrases)
        
        return updated_model, updated_tokenizer

    raise ValueError


def extend_model(model, new_phrase_weights):
    """ Modify an AutoModelForCausalLM to decode additional strings 
    
    Arguments:
        model (AutoModelForCausalLM): base model
        new_phrase_weights (torch tensor, v x d_model): set of stacked weight tensors
        
    """

    # Uncouple encoder/decoder if necessary
    if model.config.tie_word_embeddings:
        model.config.tie_word_embeddings = False
        model.get_output_embeddings().weight = torch.nn.Parameter(
            model.get_output_embeddings().weight.data.clone()
        )

    # Append new weights to the unembedding matrix
    lm_head = model.get_output_embeddings()
    new_phrase_weights = new_phrase_weights.to(lm_head.weight)
    new_weight = torch.cat([lm_head.weight, new_phrase_weights], dim=0)
    lm_head.weight = torch.nn.Parameter(new_weight)

    # Update config
    model.config.vocab_size = new_weight.shape[0]

    return model

class DecodeExtendedTokenizer:
    """ Wrap a tokenizer to decode on an extended vocab (and change nothing else) """

    def __init__(self, base_tokenizer, new_token_strs):
        self._tok = base_tokenizer
        self._extra_lookup = {len(base_tokenizer) + i: s for i, s in enumerate(new_token_strs)}

    def decode(self, ids, **kwargs):
        ids = list(ids)
        # fast path: jlens calls tok.decode([t]) with a single id at a time
        if len(ids) == 1 and ids[0] in self._extra_lookup:
            return self._extra_lookup[ids[0]]
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