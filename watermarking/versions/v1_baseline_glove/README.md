# v1 — baseline word-level (GloVe contextual)

**Idea:** Word-level red/green watermarking. Anchors are selected by hashing;
substitution candidates come from GloVe embeddings + a BERT masked-LM fill.

**Interface:**
- `apply_watermark_v2(text_list, K, tau, v_dict)` → `(tokens, stats)`
- `detect_watermark(text_list, K, tau, v_dict)` → z-score

**Config:** `SECRET_KEY`, `TAU`, `GLOVE_PATH` at the top of `watermark.py`.
`GLOVE_PATH` resolves relative to the repo root; override with the `GLOVE_PATH`
env var. Requires the 300-d GloVe vectors file (not in the repo).

**Attacks:** uses the shared `watermarking/common/attack.py`.

**Run:** `python -m experiments.run --version v1`
