# v3 — TF-IDF + LexRank word families

**Idea:** Pick the "main" sentence with TF-IDF + LexRank, build word families
around its content words, then apply a hash-based red/green rule to family
members in the other sentences. Pure CPU (nltk only), no ML models.

**Interface:**
- `watermark(text)` → dict (`watermarked_text`, `families`, `changes`, …)
- `detect(text, families=None)` → dict (`z_score`, `green_fraction`, …)

**Attacks:** uses the shared `watermarking/common/attack.py`.

**Run:** `python -m experiments.run --version v3`
