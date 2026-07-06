# v2 — sentence-level (macro-semantic)

**Idea:** Watermark at the sentence level. Sentence embeddings
(SentenceTransformer) decide "anchor" sentences; non-anchors are paraphrased
(T5) toward a hash-selected green class. No GloVe file needed.

**Interface:**
- `apply_sentence_watermark(text, K)` → `(text, stats)`
- `detect_sentence_watermark(text, K)` → z-score

**Config:** `SECRET_KEY` at the top of `sentence_watermark.py`.

**Attacks:** this version keeps its **own** `attack.py` (a divergent, trimmed
copy). It now includes the active/passive transform (`syn_transform`), which
imports `active_to_passive` from the shared `common/act_pas` library. It still
lacks `paraphrasing_attack` / `reorder_random` that `common/attack.py` has.
TODO: fully reconcile the two attack modules.

**Run:** `python -m experiments.run --version v2`
