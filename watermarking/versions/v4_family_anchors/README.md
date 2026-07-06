# v4 — family-based context anchors

**Idea:** Successor to v3. Main sentence chosen by TextRank over sentence
embeddings; word families filtered by a BERT masked-LM so substitutions stay
context-appropriate; hash-based red/green over anchored regions.

**Interface:**
- `watermark(text)` → dict (`watermarked_text`, `anchor_roots`,
  `context_families`, `changes`, …)
- `detect(text, context_families=None)` → dict (`z_score`, `green_fraction`, …)

**Attacks:** uses the shared `watermarking/common/attack.py`.

**Run:** `python -m experiments.run --version v4`
