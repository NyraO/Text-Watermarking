# Text Watermarking

This repository contains an experimental implementation of an anchored-based text watermarking system and a collection of attacks to evaluate watermark robustness. The project includes:

- `watermark.py` - core encoder/decoder logic and utilities (tokenization, embedding loading, anchor selection, watermark application and detection).
- `attack.py` - a set of text manipulation attacks (deletion, insertion, paraphrasing, synonym substitution, reordering, noise injection, etc.).
- `experiment_runner.py` - example experiment simulator that applies the watermark to  texts and runs attacks to report detection scores.

 
## Getting started

Prerequisites

- A copy of a 300-dimensional word embedding file (GloVe-style plain text). The project expects the file path to be configured in `watermark.py` via the `GLOVE_PATH` variable.


 
You can install typical dependencies with pip. 
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy nltk transformers torch
```

NLTK data

The first run of `watermark.py` attempts to download WordNet if needed. You can also download it manually:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

 
## Quick usage

Run the example experiment runner which applies the watermark to `bbc.txt` and runs a set of attacks:

```bash
python experiment_runner.py
```


 

