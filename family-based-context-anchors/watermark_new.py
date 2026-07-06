import math
import hashlib
import string
import numpy as np
import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.utils import logging as hf_logging

# Quiet the "loading weights / some weights were not used" messages
hf_logging.set_verbosity_error()
# Disable the "Loading weights: 100%|...|" tqdm progress bars
hf_logging.disable_progress_bar()

_bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
_bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
_bert_model.eval()

def _bert_filter(sentence: str, word: str, candidates: set[str], keep: int = 5) -> set[str]:
    """using BERT to rank WordNet candidates in context."""
    if not candidates:
        return candidates

    masked = sentence.replace(word, _bert_tok.mask_token, 1)
    if _bert_tok.mask_token not in masked:        # word not found verbatim
        return candidates

    inputs = _bert_tok(masked, return_tensors="pt")
    mask_pos = (inputs.input_ids == _bert_tok.mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_pos) == 0:
        return candidates

    with torch.no_grad():
        logits = _bert_model(**inputs).logits[0, mask_pos[0], :]
    probs = torch.softmax(logits, dim=-1)

    scored = []
    for cand in candidates:
        ids = _bert_tok.encode(cand, add_special_tokens=False)
        if len(ids) != 1:                         # skip multi-token candidates
            continue
        scored.append((cand, probs[ids[0]].item()))

    if not scored:
        return candidates                         # nothing single-token; fall back
    scored.sort(key=lambda x: x[1], reverse=True)
    return {c for c, _ in scored[:keep]}

def _ensure_nltk():
    for pkg in ("punkt", "averaged_perceptron_tagger", "wordnet", "stopwords",
                "punkt_tab", "averaged_perceptron_tagger_eng"):
        for prefix in ("tokenizers", "taggers", "corpora"):
            try:
                nltk.data.find(f"{prefix}/{pkg}")
                break
            except LookupError:
                pass
        else:
            nltk.download(pkg, quiet=True)


_ensure_nltk()

STOPWORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def is_content_word(token: str, tag: str) -> bool:
    return (
        tag.startswith(("NN", "VB", "JJ", "RB"))
        and not tag.startswith(("NNP", "NNPS"))
        and token.lower() not in STOPWORDS
        and token not in PUNCT
        and token.isalpha()
    )


def wordnet_pos(tag: str):
    if tag.startswith("NN"):
        return wn.NOUN
    if tag.startswith("VB"):
        return wn.VERB
    if tag.startswith("JJ"):
        return wn.ADJ
    if tag.startswith("RB"):
        return wn.ADV
    return None


def tokens_to_text(tokens: list[str]) -> str:
    text = ""
    for tok in tokens:
        if tok in PUNCT or (text and text[-1] in '("'):
            text += tok
        else:
            text += (" " if text else "") + tok
    return text


def get_synonyms(word: str, wn_pos) -> set[str]:
    """Return a set of synonyms for word"""
    syns = set()
    for synset in wn.synsets(word, pos=wn_pos):
        for lemma in synset.lemmas():
            original = lemma.name()
            if original[0].isupper():  # Skip capitalised lemmas — they are proper nouns
                continue
            candidate = original.replace("_", " ").lower()
            if candidate != word.lower() and " " not in candidate and candidate.isalpha():
                syns.add(candidate)
    return syns


def build_family(word: str, tag: str) -> set[str] | None:
    if not is_content_word(word, tag):
        return None
    wn_pos = wordnet_pos(tag)
    if wn_pos is None:
        return None
    syns = get_synonyms(word.lower(), wn_pos)
    if not syns:
        return None
    return syns | {word.lower()}


def embed_sentences(sentences):
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    return enc.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity_matrix(embeddings):
    return embeddings @ embeddings.T


def textrank(sim_matrix: np.ndarray, damping: float = 0.85,
             max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Power-iteration TextRank over a similarity matrix
    The similarity matrix is row-normalised to form a stochastic transition
    matrix, then the standard PageRank recurrence is applied
    """
    n = sim_matrix.shape[0]

    # Zero the diagonal (a sentence is not similar to itself for ranking)
    A = sim_matrix.copy()
    np.fill_diagonal(A, 0.0)

    # Clip negatives (cosine similarity can be slightly negative)
    A = np.clip(A, 0.0, None)

    # Row-normalise; handle all-zero rows with uniform fallback
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A = A / row_sums

    # Power iteration
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * A.T @ scores
        if np.abs(new_scores - scores).max() < tol:
            break
        scores = new_scores
    return new_scores


def find_main_sentence(sentences: list[str]) -> tuple[int, str]:
    """
    Select the sentence that is most central to the whole document
    Steps:
      1. Embed sentences with BERT
      2. Compute pairwise cosine similarity
      3. Run TextRank
    Return the index and text of the highest-scoring sentence
    """
    embeddings = embed_sentences(sentences)
    sim = cosine_similarity_matrix(embeddings)
    scores = textrank(sim)
    idx = int(np.argmax(scores))
    return idx, sentences[idx]


def build_families_for_sentences(sentences: list[str]) -> dict[str, set[str]]:
    """Build synonym families for all eligible content words across *sentences*."""
    families: dict[str, set[str]] = {}
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        for token, tag in tagged:
            root = token.lower()
            if root in families:
                continue
            fam = build_family(token, tag)
            if fam is None:
                continue
            ranked = _bert_filter(sent, token, fam - {root}, keep=5)
            families[root] = ranked | {root}
    return families


def find_anchor_roots(main_sentence: str) -> set[str]:
    """
    Return the set of root forms (lower-case) for every content word in the
    main sentence that has a non-empty synonym family. These roots define
    which words in other sentences are *anchor words*
    """
    main_families = build_families_for_sentences([main_sentence])
    return set(main_families.keys())


def identify_anchors_in_sentence(tagged: list[tuple[str, str]], anchor_roots: set[str]) -> list[int]:
    """
    Return the indices (into *tokens*) of anchor words in a tokenised sentence
    """
    anchors = []
    for i, (tok, tag) in enumerate(tagged):
        if tok.lower() in anchor_roots and is_content_word(tok, tag):
            anchors.append(i)
    return anchors


def is_green(chain_head: str, candidate: str) -> bool:
    key = (chain_head.lower() + "|" + candidate.lower()).encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % 2 == 0


def rewrite_sentence(sentence: str,
                     anchor_roots: set[str],
                     context_families: dict[str, set[str]]
                     ) -> tuple[str, list[tuple[str, str]]]:
    """
    Rewrite one non-main sentence using chain encoding

    Returns the rewritten sentence and a list of (original, replacement) pairs
    """
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    result = list(tokens)
    changes: list[tuple[str, str]] = []
    chain_head: str | None = None

    for i, (tok, tag) in enumerate(tagged):
        root = tok.lower()

        if not is_content_word(tok, tag):
            continue

        if root in anchor_roots:
            # Anchor word: reset chain head, do NOT substitute
            chain_head = root
            continue

        # Context (non-anchor) content word
        if chain_head is None:
            # No anchor seen yet in this sentence; just update chain head
            chain_head = root
            continue

        if root in context_families:
            members = context_families[root]
            greens = {m for m in members if is_green(chain_head, m)}

            if greens and not is_green(chain_head, root):
                # Current word is red and a green substitute exists
                replacement = sorted(greens)[0]
                if tok[0].isupper():
                    replacement = replacement.capitalize()
                result[i] = replacement
                changes.append((tok, replacement))
                chain_head = replacement.lower()
            else:
                # Already green, or no green substitute available
                chain_head = root
        else:
            chain_head = root

    return tokens_to_text(result), changes


def watermark(text: str) -> dict:
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        raise ValueError("Text must contain at least 2 sentences.")

    # 1. Find main/anchor sentence via BERT + TextRank
    main_idx, main_sent = find_main_sentence(sentences)

    # 2. Identify anchor roots from the main sentence
    anchor_roots = find_anchor_roots(main_sent)
    if not anchor_roots:
        raise ValueError("Main sentence has no content words with synonym families.")

    # 3. Build context families for all other sentences
    other_sentences = [s for i, s in enumerate(sentences) if i != main_idx]
    context_families = build_families_for_sentences(other_sentences)

    # 4. Rewrite non-main sentences
    watermarked_sentences = list(sentences)
    all_changes: list[tuple[int, str, str]] = []

    for s_idx, sent in enumerate(sentences):
        if s_idx == main_idx:
            continue
        new_sent, changes = rewrite_sentence(sent, anchor_roots, context_families)
        watermarked_sentences[s_idx] = new_sent
        for orig, new in changes:
            all_changes.append((s_idx, orig, new))

    return {
        "watermarked_text": " ".join(watermarked_sentences),
        "main_sentence_index": main_idx,
        "main_sentence": main_sent,
        "anchor_roots": anchor_roots,
        "context_families": context_families,
        "changes": all_changes,
    }


def detect(text: str, context_families: dict[str, set[str]] | None = None) -> dict:

    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return {"is_watermarked": False, "reason": "too short"}

    main_idx, _ = find_main_sentence(sentences)

    anchor_roots = find_anchor_roots(_)
    if not anchor_roots:
        return {"is_watermarked": False, "reason": "no anchor roots"}

    if context_families is None:
        other = [s for i, s in enumerate(sentences) if i != main_idx]
        context_families = build_families_for_sentences(other)

    if not context_families:
        return {"is_watermarked": False, "reason": "no context families"}

    total_tested = 0
    total_green = 0
    details: list[dict] = []

    for s_idx, sent in enumerate(sentences):
        if s_idx == main_idx:
            continue

        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        chain_head: str | None = None

        for tok, tag in tagged:
            root = tok.lower()
            if not is_content_word(tok, tag):
                continue

            if root in anchor_roots:
                chain_head = root
                continue

            if chain_head is not None and root in context_families:
                green = is_green(chain_head, root)
                total_tested += 1
                if green:
                    total_green += 1
                details.append({
                    "sentence": s_idx,
                    "chain_head": chain_head,
                    "token": tok,
                    "green": green,
                })

            # Advance chain head even if word not in families
            if is_content_word(tok, tag):
                chain_head = root

    if total_tested == 0:
        return {
            "is_watermarked": False,
            "reason": "no testable tokens found",
            "tested_tokens": 0,
        }

    match_fraction = total_green / total_tested
    p0 = 0.5
    sd = math.sqrt(p0 * (1 - p0) / total_tested)
    z = (match_fraction - p0) / sd if sd > 0 else 0.0
    is_watermarked = z > 4.0 and match_fraction >= 0.5

    return {
        "is_watermarked": is_watermarked,
        "green_fraction": round(match_fraction, 4),
        "tested_tokens": total_tested,
        "green_tokens": total_green,
        "z_score": round(z, 4),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("text.txt") as f:
        DEMO_TEXT = f.read().strip()

    SEP = "=" * 70

    print(SEP)
    print("ORIGINAL TEXT")
    print(SEP)
    print(DEMO_TEXT)
    print()

    result = watermark(DEMO_TEXT)

    print()
    print(SEP)
    print("WATERMARKED TEXT")
    print(SEP)
    print(result["watermarked_text"])
    print()
    print(f"Main sentence (idx={result['main_sentence_index']}):")
    print(f"  {result['main_sentence']!r}")
    print()
    print(f"Anchor roots ({len(result['anchor_roots'])}):")
    print(f"  {sorted(result['anchor_roots'])}")
    print()
    print(f"Context families ({len(result['context_families'])}):")
    for root in sorted(result['context_families']):
        print(f"  {root!r:12s} -> {sorted(result['context_families'][root])}")
    print()
    print("Token substitutions applied:")
    if result["changes"]:
        for s_idx, orig, new in result["changes"]:
            print(f"  sentence {s_idx}: {orig!r} -> {new!r}")
    else:
        print("  (none — all eligible tokens were already green)")
    print()

    print(SEP)
    print("DETECTION — original (unwatermarked) text")
    print(SEP)
    det_orig = detect(
        DEMO_TEXT
    )
    for k, v in det_orig.items():
        if k != "details":
            print(f"  {k}: {v}")
    print()

    print(SEP)
    print("DETECTION — watermarked text")
    print(SEP)
    det = detect(
        result["watermarked_text"]
    )
    for k, v in det.items():
        if k != "details":
            print(f"  {k}: {v}")

    print(find_main_sentence(sent_tokenize(result["watermarked_text"])))
