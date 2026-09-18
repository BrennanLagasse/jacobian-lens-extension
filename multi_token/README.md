## Multi-Token Extension of J-Lens

### Setup
Please see the README in jlens, setup should be the same

Use ```Qwen/Qwen3.5-9B-Base``` with the Hub lens: the only 9B lens on ```neuronpedia/jacobian-lens``` is fitted on the Base model's activations, and a lens fitted on one model does not transfer to another. (Or fit a lens on the post-trained model with ```jlens.fit```.)

### File Setup

* ```phrase_context_miner.py``` mines sample usages of multi-token phrases from a pretrain dataset of your choice
* ```collect_embeddings.py``` computes the final representations preceeding each use of the target phrases
* ```process_representations.py``` computes the average representation for a given phrase
* ```embed_baseline.py``` computes the average final representation across a wide variety of text inputs
* ```fit_phrase_rows.py``` fits calibrated decoding rows for the phrases by logistic regression (see below)
* ```extend_model.py``` generates a model that can decode to an extended vocab including new phrases and a tokenizer that can decode these outputs
* ```walkthrough_multitoken.ipynb``` outlines the core experimental logic and visualizes results

### Extended Model Options

There are a few different approaches included for picking weights assigned with new tokens that represent multi-token phrases (see options in ```extend_model.py```). Check the code for exact details as this may slightly shift.
* FITTED_ROWS (**recommended**): rows fitted by ```fit_phrase_rows.py``` — bias-free logistic regression over the extended softmax, positives (state before the phrase) weighted to the corpus prior, negatives = generic FineWeb positions, L-BFGS. The prototype methods below have the right norm but ~10x the logit variance of a real ```W_U``` row, so they land in the top-10 at ~19% of generic positions and fill every lens readout regardless of context; the fitted rows put mass on the new tokens equal to the corpus prior, generic top-10 rate ~0, AUROC 0.98, and rank the phrase within ~1.25x of its real first token's rank (measured on Qwen3.5-9B-Base, 15 phrases, 2,942 held-out positions). The script prints these held-out numbers for every row; check them before reading any lens plot.
* PRIOR_REPRESENTATION_EMBED: compute the decoding weights for a phrase as the unit pointing in the direction of the average final representation preceeding the first word computed over a variety of texts.
* AVERAGE_TOKEN_WEIGHTS: compute the decoding weights for a phrase as an average of its component tokens

