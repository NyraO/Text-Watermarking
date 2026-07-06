import os
import math
import random
import re
from nltk.corpus import wordnet as wn
from nltk import pos_tag

from random import randrange
from nltk import sent_tokenize
# Ensure you have your active/passive library installed/accessible
try:
    from act_pas_lib.act_pas import active_to_passive
except ImportError:
    pass # Will gracefully fail in the runner

def tokenizer(text):
    """Splits text into words and punctuation marks."""
    return re.findall(r'\w+|[^\w\s]', text)

def rebuild_text(tokens):
    """Joins tokens but removes the weird spaces before commas/periods."""
    if isinstance(tokens, str):
        return tokens
    text = " ".join(tokens)
    return re.sub(r'\s+([^\w\s])', r'\1', text)

def copy_paste_attack(watermarked_text, excerpt, dilution_rate, position):
    wm_tokens = tokenizer(watermarked_text)
    excerpt_tokens = tokenizer(excerpt)

    size_watermarked = len(wm_tokens)
    total_size_needed = math.ceil(size_watermarked / dilution_rate)
    size_excerpt_needed = total_size_needed - size_watermarked

    cropped_tokens = excerpt_tokens[:size_excerpt_needed]
    before_tokens = cropped_tokens[:position]
    after_tokens = cropped_tokens[position:] 

    return rebuild_text(before_tokens + wm_tokens + after_tokens)

def insertion_attack(watermarked_text, ratio, words):
    tokens = tokenizer(watermarked_text)
    num_words_to_insert = math.ceil(len(tokens) * ratio)
    
    # Sort positions in REVERSE order so insertions don't shift upcoming targets
    random_positions = sorted([random.randint(0, len(tokens)) for _ in range(num_words_to_insert)], reverse=True)

    for pos in random_positions:
        tokens.insert(pos, random.choice(words))

    return rebuild_text(tokens)

def insert_noise_attack(watermarked_text, ratio, punctuations=[",", ":", ";"]):
    tokens = tokenizer(watermarked_text)
    num_words_to_modify = math.ceil(len(tokens) * ratio)
    random_positions = random.sample(range(len(tokens)), min(num_words_to_modify, len(tokens)))

    for pos in random_positions:
        mod_type = random.randint(1, 3)
        if mod_type == 1:
            tokens[pos] += random.choice(punctuations)
        elif mod_type == 2 and len(tokens[pos]) > 1:
            word = tokens[pos]
            idx = random.randrange(0, len(word))
            tokens[pos] = word[:idx] + word[idx + 1:]
        elif mod_type == 3 and len(tokens[pos]) > 0:
            word = tokens[pos]
            idx = random.randrange(0, len(word) + 1)
            tokens[pos] = word[:idx] + random.choice(["a", "e", "i", "o", "u"]) + word[idx:]

    return rebuild_text(tokens)

def deletion(text, strength):
    tokens = tokenizer(text) 
    output_tokens = [token for i, token in enumerate(tokens) if (i % strength) != 0]
    return rebuild_text(output_tokens)

def delete_random(text, strength):
    tokens = tokenizer(text)
    # Ensure we don't try to delete more tokens than exist
    strength = min(strength, len(tokens))
    delete_indices = set(random.sample(range(len(tokens)), strength))
    
    output_tokens = [word for i, word in enumerate(tokens) if i not in delete_indices]
    return rebuild_text(output_tokens)

def delete_portion(text, begin, end):
    tokens = tokenizer(text)
    # Replaced splitlines logic with direct token slicing for consistency
    output_tokens = tokens[:begin] + tokens[end:]
    return rebuild_text(output_tokens)

def generative_attack(text, token, n):
    tokens = tokenizer(text)
    output_words = []
    for i in range(len(tokens)):
        output_words.append(tokens[i])
        if i % n == 0 and i != 0:
            output_words.append(token)
    return rebuild_text(output_words)

# def paraphrasing_attack(text, style, temperature=0.6, top_p=0.6, size=1):
#     # SECURE API KEY HANDLING
#     key = os.environ.get("GROQ_API_KEY")
#     if not key:
#         raise ValueError("GROQ_API_KEY environment variable not set.")

#     task = f"""
#     You are a paraphrasing engine. Rewrite the user's text while preserving meaning.
#     Style: {style}.
#     Rules:
#     - Do not shorten meaning
#     - Do not add new ideas
#     - Produce natural and fluent English
#     """

#     client = Groq(api_key=key)
#     res = ""
#     completion = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": task},
#             {"role": "user", "content": text}
#         ],
#         temperature=temperature,
#         max_tokens=int(sum([1 for c in text if c.isalpha()]) * 0.25 * size),
#         top_p=top_p,
#         stream=True,
#         stop=None
#     )
#     for chunk in completion:
#         res += chunk.choices[0].delta.content or ""
#     return res

def synonym_attack(text, replace_prob=0.2, max_replace_ratio=0.1, seed=None):
    if seed is not None: random.seed(seed)
    tokens = tokenizer(text)
    words_only = [t for t in tokens if t.isalpha()]
    if not words_only: return text
    
    pos_tags = pos_tag(words_only)
    token_word_indices = [i for i, t in enumerate(tokens) if t.isalpha()]
    max_replacements = int(len(words_only) * max_replace_ratio)
    
    candidate_positions = list(range(0, len(words_only), max(1, len(words_only) // max_replacements)))[:max_replacements]
    replaced = 0
    
    for wpos in candidate_positions:
        if replaced >= max_replacements: break
        if random.random() > replace_prob: continue
        
        word, penn_pos = pos_tags[wpos]
        wn_pos = penn_to_wn_pos(penn_pos)
        if wn_pos is None: continue
        
        synonym = get_synonym(word, wn_pos)
        if synonym and synonym.lower() != word.lower():
            token_idx = token_word_indices[wpos]
            tokens[token_idx] = synonym.upper() if word.isupper() else synonym.lower()
            replaced += 1
            
    return rebuild_text(tokens)

def penn_to_wn_pos(penn_pos: str):
    if penn_pos.startswith("N"): return wn.NOUN
    if penn_pos.startswith("V"): return wn.VERB
    if penn_pos.startswith("J"): return wn.ADJ
    if penn_pos.startswith("R"): return wn.ADV
    return None

def get_synonym(word, wn_pos):
    synonyms = wn.synsets(word, pos=wn_pos)
    if not synonyms: return None
    candidates = [lemma.name().replace("_", " ") for syn in synonyms for lemma in syn.lemmas() if lemma.name().lower() != word.lower()]
    return random.choice(candidates) if candidates else None

def reorder(text, strength, distance):
    tokens = tokenizer(text)
    output_words = []
    count = 0
    marked = {}

    for word in tokens:
        if (count % strength) == 0:
            marked[count] = word
        else:
            output_words.append(word)

        for c in list(marked.keys()):
            if (c + distance) == count:
                output_words.append(marked[c])
                marked.pop(c)
        count += 1

    for c in sorted(marked.keys()):
        output_words.append(marked[c])

    return rebuild_text(output_words)

def reorder_random_max_dist(text, strength, max_distance):
    tokens = tokenizer(text)
    output_words = []
    count = 0
    marked = {}

    strength = min(strength, len(tokens))
    rand_indices = set(random.sample(range(len(tokens)), strength))

    for word in tokens:
        if count in rand_indices:
            marked[count + randrange(1, max_distance + 1)] = word
        else:
            output_words.append(word)

        for c in list(marked.keys()):
            if c == count:
                output_words.append(marked[c])
                marked.pop(c)
        count += 1

    for c in sorted(marked.keys()):
        output_words.append(marked[c])
    
    return rebuild_text(output_words)

def syn_transform(text, strength):
    sentences = sent_tokenize(text)
    output_sentences = []
    
    for count, s in enumerate(sentences):
        if (count % strength) == 0:
            s_out = active_to_passive(s)
        else:
            s_out = s
            
        s_out = re.sub(r'<.!?>', '', s_out).strip() + ' '
        output_sentences.append(s_out)

    return " ".join(output_sentences)