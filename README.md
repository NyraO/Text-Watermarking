# Text Watermarking

Experimental anchor-based text watermarking, with a set of attacks for
evaluating robustness. The repo holds **several versions of the watermarking
algorithm as sibling packages** so they can be run and compared side by side.

## Layout

```
watermarking/
├── common/                     # shared across all versions
│   ├── attack.py               # deletion / insertion / reorder / synonym / active-passive …
│   └── act_pas/                # active↔passive transformation library
└── versions/
    ├── v1_baseline_glove/      # word-level, GloVe contextual candidates
    ├── v2_sentence/            # sentence-level, SentenceTransformer + T5 (own attack.py)
    ├── v3_tfidf/               # TF-IDF + LexRank word families 
    └── v4_family_anchors/      # BERT-filtered families + TextRank anchors

experiments/run.py              # single entry point; dispatches to any version
data/inputs/                    # input corpus (bbc.txt, goethe.txt, text.txt)
results/                        # generated outputs (git-ignored)
```

Each version exposes its own `watermark`/`detect` (interfaces differ) and a
`runner.py` benchmark. See each version's `README.md` for its interface and
notes.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy nltk transformers torch sentence-transformers sumy openai
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

`v1` additionally needs a 300-d GloVe-style vectors file; set its location with
the `GLOVE_PATH` env var (see `watermarking/versions/v1_baseline_glove/README.md`).

### Optional: LLM paraphrasing attack

The benchmark's paraphrasing attack (attack #10 in every runner) rewrites the
watermarked text with a local LLM served by [Ollama](https://ollama.com) at
`http://localhost:11434`. Start the server and pull the model once:

```bash
ollama serve &
ollama pull llama3.1:8b
```

This is optional — if Ollama is not running, the runners print
`10. LLM Paraphrasing SKIPPED: ...` and continue with the other attacks.

## Running

Always run from the repo root using the module form so package imports resolve:

```bash
python -m experiments.run --list                 # show all versions
python -m experiments.run --version v3            # run one on its default input
python -m experiments.run --version v1 --input data/inputs/goethe.txt
```

You can also run a version's runner directly, e.g.
`python -m watermarking.versions.v3_tfidf.runner`.

## Adding a new version

1. Copy an existing folder under `watermarking/versions/` to `vN_<name>/`.
2. Keep `watermark` / `detect` (or your own entry points) and a `runner.py`.
3. Register it in `experiments/run.py` (`VERSIONS` + `DESCRIPTIONS`).
4. Reuse `watermarking/common/attack.py` rather than copying attacks.
5. Add a short `README.md` saying what changed and why the version exists.
