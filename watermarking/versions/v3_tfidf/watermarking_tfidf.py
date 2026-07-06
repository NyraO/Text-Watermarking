import re
import math
import hashlib
import string
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def ensure_nltk():
    for pkg in ("punkt", "averaged_perceptron_tagger", "wordnet", "stopwords",
                "punkt_tab", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{pkg}")
            except LookupError:
                try:
                    nltk.data.find(f"corpora/{pkg}")
                except LookupError:
                    nltk.download(pkg, quiet=True)


ensure_nltk()

STOPWORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def is_content_word(token, tag):
    return (
            tag.startswith(("NN", "VB", "JJ", "RB"))
            and token.lower() not in STOPWORDS
            and token not in PUNCT
            and token.isalpha()
    )


def wordnet_pos(tag):
    """Map Penn tag to WordNet POS."""
    if tag.startswith("NN"):
        return wn.NOUN
    if tag.startswith("VB"):
        return wn.VERB
    if tag.startswith("JJ"):
        return wn.ADJ
    if tag.startswith("RB"):
        return wn.ADV
    return None


def synonyms(word, wn_pos):
    """Return a set of synonyms for word"""
    syns = set()
    for synset in wn.synsets(word, pos=wn_pos):
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word.lower() and " " not in candidate:
                syns.add(candidate)
    return syns


def idf_calculate(word, sentences):
    """Inverse document frequency of word across a list of token lists"""
    n = len(sentences)
    df = sum(1 for s in sentences if word.lower() in {t.lower() for t in s})
    return math.log((n + 1) / (df + 1)) + 1


def tfidf_score(sentence_tokens, all_sentences_tokens):
    """Mean TF-IDF score for content words in sentence_tokens"""
    tf_counts = Counter(t.lower() for t in sentence_tokens if t.isalpha())
    total = sum(tf_counts.values()) or 1
    scores = []
    for word, cnt in tf_counts.items():
        tf = cnt / total
        idf = idf_calculate(word, all_sentences_tokens)
        scores.append(tf * idf)
    return sum(scores) / len(scores) if scores else 0.0


def hash_green(anchor_word: str, candidate: str) -> bool:
    """Return True if candidate is on the green list for anchor_word"""
    key = (anchor_word.lower() + "|" + candidate.lower()).encode()
    digest = int(hashlib.sha256(key).hexdigest(), 16)
    return digest % 2 == 0


class WordFamily:
    """A set of synonymous words anchored to a content word"""

    def __init__(self, root: str, members: set[str]):
        self.root = root.lower()
        self.members: set[str] = {m.lower() for m in members} | {self.root}

    def contains(self, word: str) -> bool:
        return word.lower() in self.members

    def green_synonyms(self, anchor_word: str) -> set[str]:
        """Synonyms of anchor_word that are on the green list"""
        return {m for m in self.members if hash_green(anchor_word, m)}

    def red_synonyms(self, anchor_word: str) -> set[str]:
        return self.members - self.green_synonyms(anchor_word)

    # def __repr__(self):
    #     return f"WordFamily(root={self.root!r}, members={self.members})"


def _sentence_tfidf_vectors(sentences):
    """One sparse TF-IDF vector (dict word->weight) per sentence, content words only."""
    tokenized = [word_tokenize(s) for s in sentences]
    vectors = []
    for tok in tokenized:
        tf = Counter(t.lower() for t in tok
                     if t.isalpha() and t.lower() not in STOPWORDS)
        total = sum(tf.values()) or 1
        vectors.append({w: (c / total) * idf_calculate(w, tokenized)
                        for w, c in tf.items()})
    return vectors


def _cosine(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    num = sum(a[w] * b[w] for w in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def lexrank_scores(sentences, threshold=0.1, damping=0.85,
                   epsilon=1e-6, max_iter=100):
    """LexRank centrality score for every sentence (returns a list aligned to `sentences`)."""
    n = len(sentences)
    if n == 1:
        return [1.0]

    vectors = _sentence_tfidf_vectors(sentences)
    # symmetric idf-cosine similarity, thresholded (continuous LexRank)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = _cosine(vectors[i], vectors[j])
            if c >= threshold:
                sim[i][j] = sim[j][i] = c

    # PageRank power iteration on the row-normalized similarity graph
    scores = [1.0 / n] * n
    for _ in range(max_iter):
        new = [(1 - damping) / n] * n
        for i in range(n):
            row_sum = sum(sim[i])
            if row_sum == 0:                       # dangling sentence
                for j in range(n):
                    new[j] += damping * scores[i] / n
            else:
                for j in range(n):
                    if sim[i][j] > 0:
                        new[j] += damping * scores[i] * sim[i][j] / row_sum
        if sum(abs(new[k] - scores[k]) for k in range(n)) < epsilon:
            scores = new
            break
        scores = new
    return scores


def find_main_sentence(sentences: list[str]) -> tuple[int, str]:
    """Return (index, sentence) of the most central sentence by LexRank."""
    scores = lexrank_scores(sentences)
    idx = max(range(len(scores)), key=lambda i: scores[i])
    return idx, sentences[idx]


def build_families(main_sentence: str) -> list[WordFamily]:
    tokens = word_tokenize(main_sentence)
    tagged = pos_tag(tokens)
    families = []
    seen_roots = set()

    for token, tag in tagged:
        if not is_content_word(token, tag):
            continue
        root = token.lower()
        if root in seen_roots:
            continue
        seen_roots.add(root)

        wn_pos = wordnet_pos(tag)
        if wn_pos is None:
            continue
        syns = synonyms(root, wn_pos)
        if syns:  # only keep words that actually have synonyms
            families.append(WordFamily(root=root, members=syns))

    return families


def find_anchors_in_sentence(sentence: str, families: list[WordFamily]):
    tokens = word_tokenize(sentence)
    anchors = []
    for i, tok in enumerate(tokens):
        for fam in families:
            if fam.contains(tok):
                anchors.append((i, tok, fam))
                break  # one family per token
    return tokens, anchors


def apply_redgreen(tokens: list[str], anchors: list[tuple], families: list[WordFamily]) -> list[str]:
    """
    Between consecutive anchors (one section at a time), apply the red/green rule:
    1. Scan the section for any token that *must* be used (belongs to a family)
       and is on the red list for the preceding anchor.
    2. If such a "forced-red" token exists then make the section RED: replace any
       green-list tokens in the section with a red synonym so the signal is red
    3. If no forced-red token exists then make the section GREEN: replace any
       red-list tokens with a green synonym (original behaviour)
    """
    if not anchors:
        return tokens

    result = list(tokens)
    segments = []
    for k, (anchor_idx, anchor_tok, anchor_fam) in enumerate(anchors):
        start = anchor_idx + 1
        end = anchors[k + 1][0] if k + 1 < len(anchors) else len(tokens)
        segments.append((start, end, anchor_tok, anchor_fam))

    tagged_all = pos_tag(tokens)

    for start, end, anchor_tok, anchor_fam in segments:
        # A section is RED if any family token in it is red-listed for the anchor
        force_red = False
        for i in range(start, end):
            tok = tokens[i]
            if not tok.isalpha():
                continue
            for fam in families:
                if fam.contains(tok):
                    if tok.lower() in anchor_fam.red_synonyms(anchor_tok):
                        force_red = True
                    break
            if force_red:
                break

        # Rewrite tokens to match the chosen colour
        for i in range(start, end):
            tok = tokens[i]
            if not tok.isalpha():
                continue
            tag = tagged_all[i][1] if i < len(tagged_all) else ""
            if wordnet_pos(tag) is None:
                continue
            for fam in families:
                if not fam.contains(tok):
                    continue
                greens = anchor_fam.green_synonyms(anchor_tok) & fam.members
                reds = anchor_fam.red_synonyms(anchor_tok) & fam.members
                if force_red:
                    # Section should be RED: swap green tokens to red
                    if tok.lower() in greens and reds:
                        replacement = sorted(reds)[0]
                        if tok[0].isupper():
                            replacement = replacement.capitalize()
                        result[i] = replacement
                else:
                    # Section should be GREEN: swap red tokens to green
                    if tok.lower() in reds and greens:
                        replacement = sorted(greens)[0]
                        if tok[0].isupper():
                            replacement = replacement.capitalize()
                        result[i] = replacement
                break  # one family per token

    return result


def tokens_to_text(tokens: list[str]) -> str:
    """Naive detokenizer that handles punctuation spacing."""
    text = ""
    for tok in tokens:
        if tok in PUNCT or (text and text[-1] in "(\"-"):
            text += tok
        else:
            text += (" " if text else "") + tok
    return text


def watermark(text: str) -> dict:
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        raise ValueError("Text must have at least 2 sentences.")

    main_idx, main_sent = find_main_sentence(sentences)
    families = build_families(main_sent)

    if not families:
        raise ValueError("No content words with synonyms found in the main sentence.")

    watermarked_sentences = list(sentences)
    all_changes = []

    for s_idx, sent in enumerate(sentences):
        if s_idx == main_idx:
            continue  # leave main sentence untouched

        tokens, anchors = find_anchors_in_sentence(sent, families)
        if not anchors:
            continue

        new_tokens = apply_redgreen(tokens, anchors, families)

        # Record changes
        for i, (orig, new) in enumerate(zip(tokens, new_tokens)):
            if orig != new:
                all_changes.append((s_idx, orig, new))

        watermarked_sentences[s_idx] = tokens_to_text(new_tokens)

    return {
        "watermarked_text": " ".join(watermarked_sentences),
        "main_sentence_index": main_idx,
        "main_sentence": main_sent,
        "families": families,
        "changes": all_changes,
    }


def detect(text: str, families: list[WordFamily] | None = None) -> dict:
    # 1. Find the main sentence
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return {"is_watermarked": False, "confidence": 0.0, "reason": "too short"}

    main_idx, main_sent = find_main_sentence(sentences)

    # 2. Recover word families
    if families is None:
        families = build_families(main_sent)

    if not families:
        return {"is_watermarked": False, "confidence": 0.0, "reason": "no families"}

    # 3. Check each section
    total_tested = 0
    total_matched = 0  # tokens that are on the *expected* list for their section
    details = []

    for s_idx, sent in enumerate(sentences):
        if s_idx == main_idx:
            continue

        tokens, anchors = find_anchors_in_sentence(sent, families)
        if not anchors:
            continue

        tagged_all = pos_tag(tokens)

        for k, (anchor_idx, anchor_tok, anchor_fam) in enumerate(anchors):
            start = anchor_idx + 1
            end = anchors[k + 1][0] if k + 1 < len(anchors) else len(tokens)

            # Determine expected section colour
            section_is_red = False
            for i in range(start, end):
                tok = tokens[i]
                if not tok.isalpha():
                    continue
                for fam in families:
                    if fam.contains(tok):
                        if tok.lower() in anchor_fam.red_synonyms(anchor_tok):
                            section_is_red = True
                        break
                if section_is_red:
                    break

            # Score tokens against expected colour
            for i in range(start, end):
                tok = tokens[i]
                if not tok.isalpha():
                    continue
                tag = tagged_all[i][1] if i < len(tagged_all) else ""
                if wordnet_pos(tag) is None:
                    continue
                for fam in families:
                    if not fam.contains(tok):
                        continue
                    is_green = hash_green(anchor_tok, tok)
                    # Token matches expectation when colour agrees with section colour
                    matched = (not is_green) if section_is_red else is_green
                    total_tested += 1
                    if matched:
                        total_matched += 1
                    details.append({
                        "sentence": s_idx,
                        "anchor": anchor_tok,
                        "token": tok,
                        "section_red": section_is_red,
                        "green": is_green,
                        "matched": matched,
                    })
                    break

    if total_tested == 0:
        return {"is_watermarked": False, "confidence": 0.0,
                "reason": "no testable tokens found", "tested_tokens": 0}

    match_fraction = total_matched / total_tested
    p0 = 0.5
    se = math.sqrt(p0 * (1 - p0) / total_tested)
    z = (match_fraction - p0) / se if se > 0 else 0.0
    is_watermarked = z > 2 and match_fraction > 0.5

    return {
        "is_watermarked": is_watermarked,
        "green_fraction": round(match_fraction, 4),
        "tested_tokens": total_tested,
        "green_tokens": total_matched,
        "z_score": round(z, 4),
        "details": details,
    }


if __name__ == "__main__":
    text = (
        """Nvidia has announced a new chip for PCs as it moves into the consumer market for devices integrated with AI technology. "This reinvention of the computer is as big of a deal as the reinvention of the phone into what we now know as the smartphone," Nvidia's chief executive Jensen Huang said as he unveiled the RTX Spark chip. Huang made the announcement on Monday as he delivered a keynote speech ahead of the opening of the Computex technology show in Taipei, Taiwan. Separately on Sunday, the US tightened its rules on selling Nvidia's most advanced chips to Chinese firms. The RTX Spark is "a new superchip... for the era of personal AI agents - offering a new class of computer that moves from tool to teammate," Nvidia said on its website. It will be included in a new line of Windows PCs made by Lenovo, HP, Dell, Microsoft Surface, Asus, and MSI. They are due to be available in the autumn, with models from Acer and Gigabyte to follow. The move marks a challenge to high-profile names in the PC market like Apple and Intel. Lenovo, HP, Dell and Apple accounted for almost 75% of the world 's PC market in the first three months of this year, according to research firm Gartner.""")

    print("=" * 70)
    print("ORIGINAL TEXT")
    print("=" * 70)
    print(text)
    print()

    result = watermark(text)

    print("=" * 70)
    print("WATERMARKED TEXT")
    print("=" * 70)
    print(result["watermarked_text"] + "\n")
    print(f"Main sentence (idx={result['main_sentence_index']}): {result['main_sentence']!r}" + "\n")
    print("Families discovered:")
    for fam in result["families"]:
        print(f"  {fam.root!r:10s} -> {sorted(fam.members)}")
    print()
    print("Token changes applied:")
    if result["changes"]:
        for s_idx, orig, new in result["changes"]:
            print(f"  sentence {s_idx}: {orig!r} → {new!r}")
    else:
        print("  (none — no red-list tokens found)")
    print()

    print("=" * 70)
    print("DETECTION — original (unwatermarked) text")
    print("=" * 70)
    det_orig = detect(text, families=result["families"])
    for k, v in det_orig.items():
        if k != "details":
            print(f"  {k}: {v}")
    print()

    print("=" * 70)
    print("DETECTION — watermarked text")
    print("=" * 70)
    det = detect(result["watermarked_text"], families=result["families"])
    for k, v in det.items():
        if k != "details":
            print(f"  {k}: {v}")

