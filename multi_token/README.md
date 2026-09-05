## Multi-Token Extension of J-Lens

### Setup
Please see the README in jlens, setup should be the same

### File Setup

* ```phrase_context_miner.py``` mines sample usages of multi-token phrases from a pretrain dataset of your choice
* ```collect_embeddings.py``` computes the final representations preceeding each use of the target phrases
* ```process_representations.py``` computes the average representation for a given phrase
* ```embed_baseline.py``` computes the average final representation across a wide variety of text inputs
* ```extend_model.py``` generates a model that can decode to an extended vocab including new phrases and a tokenizer that can decode these outputs
* ```walkthrough_multitoken.ipynb``` outlines the core experimental logic and visualizes results

### Extended Model Options

There are a few different approaches included for picking weights assigned with new tokens that represent multi-token phrases (see options in ```extend_model.py```). Check the code for exact details as this may slightly shift.
* PRIOR_REPRESENTATION_EMBED: compute the decoding weights for a phrase as the unit pointing in the direction of the average final representation preceeding the first word computed over a variety of texts.
* AVERAGE_TOKEN_WEIGHTS: compute the decoding weights for a phrase as an average of its component tokens

