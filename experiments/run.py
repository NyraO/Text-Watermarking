#!/usr/bin/env python3
"""Single entry point for running any watermarking version's benchmark.

Usage:
    python -m experiments.run --version v1 --input data/inputs/bbc.txt
    python -m experiments.run --version v4          # uses that version's default input
    python -m experiments.run --list

Each version keeps its own runner (the algorithms have different interfaces);
this dispatcher just maps a short name to the right runner entry point so you
don't have to remember module paths.
"""
import argparse
import importlib
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# short name -> (runner module, entry function, default input under data/inputs/)
VERSIONS = {
    "v1": ("watermarking.versions.v1_baseline_glove.runner", "run_experiment",          "bbc.txt"),
    "v2": ("watermarking.versions.v2_sentence.runner",       "run_sentence_experiment", "bbc.txt"),
    "v3": ("watermarking.versions.v3_tfidf.runner",          "run_experiment",          "bbc.txt"),
    "v4": ("watermarking.versions.v4_family_anchors.runner", "run_experiment",          "text.txt"),
}

DESCRIPTIONS = {
    "v1": "baseline word-level, GloVe contextual candidates (apply_watermark_v2 / detect_watermark)",
    "v2": "sentence-level, SentenceTransformer + T5 paraphrase (apply_sentence_watermark / detect)",
    "v3": "TF-IDF + LexRank word families, hash red/green (watermark / detect)",
    "v4": "family-based context anchors, BERT filter + TextRank (watermark / detect)",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", choices=sorted(VERSIONS), help="which algorithm version to run")
    p.add_argument("--input", help="path to input text (default: the version's sample under data/inputs/)")
    p.add_argument("--list", action="store_true", help="list available versions and exit")
    args = p.parse_args()

    if args.list or not args.version:
        print("Available versions:")
        for name in sorted(VERSIONS):
            print(f"  {name}  - {DESCRIPTIONS[name]}")
        if not args.version:
            print("\nPass --version <name> to run one.")
        return

    module_path, func_name, default_input = VERSIONS[args.version]
    input_path = args.input or os.path.join(REPO_ROOT, "data", "inputs", default_input)

    runner = importlib.import_module(module_path)
    entry = getattr(runner, func_name)
    print(f"=== Running {args.version}: {DESCRIPTIONS[args.version]} ===")
    print(f"=== Input: {input_path} ===\n")
    entry(input_path)


if __name__ == "__main__":
    main()
