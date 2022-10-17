# CFairER
This is the implementation of CFairER (Counterfactual Explanation for Fairness in Recommendation), submitted to WWW 2023.
Counterfactual Explanation for Fairness (CFairER) is the first work that generates attribute-level counterfactual explanations for fairness from a Heterogeneous Information Network.


<!-- Reproducibility -->
## Reproducibility
0. Environment:
```
python 3.7
tensorflow 2.x
pytorch 
```
1. Create a new directory for this repo:
```
➜ mkdir CFairER
➜ cd CFairER
```
2. Get source code and dataset:
```
➜ git clone this repo
```
3. Run reinforcement learning agent:
```
➜ cd CFairER
➜ python main.py
```
4. Test and erasure-based evaluation:
```
➜ python erasure-based_evalation.py
```
<!-- Important Args -->
## Important Args
0. Edit args while run:
```
➜ cd utils
➜ vim parser.py 
```
1. Args:
```
Graph Representation module:
--n_epoch: graph embedding epoch 
--n_hid: hidden size 
--n_inp: graph embedding size # assert this with --n_factors

Recommendation model:
--n_factors: Number of latent factors (or embeddings)

CFE model (reinforce):
--ek_alpha: alpha Eq. (1)
--lambda_tradeoff: lambda Eq. (1)
--count_threshold: epsilon Eq.(9)
--embedding_size: embedding size
--rnn_size: RNN size
--erasure: erasure evaluation rate
--erasure_batch: erasure evaluation batch size
```
<!-- Cite information -->
## Cite information

```
will be released after paper acceptance.
```
