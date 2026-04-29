import os
import re
import hashlib
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from transformers import pipeline
import torch

# CONFIGURATION
SECRET_KEY = "amina_secure_key_2026"
TAU = 0.1 # 10% anchors
GLOVE_PATH = "../..//wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt" 
print("loading BERK Masked language model")
bert_fill_mask = pipeline("fill-mask", model = "bert-base-uncased")
FILES_TO_PROCESS = ["bbc.txt", "goethe.txt"]

def download_nltk_deps():
    """Downloads WordNet if not already present."""
    try:
        wordnet.ensure_loaded()
    except:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def load_glove_embeddings(file_path):
    """Reads GloVe .txt file with error handling for non-numeric values."""
    embeddings = {}
    if not os.path.exists(file_path):
        print(f"ERROR: {file_path} not found!")
        return None
    
    print(f"Loading GloVe embeddings from {file_path}... (This may take a minute)")
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            values = line.split()
            
            # Skip empty lines
            if not values:
                continue
            
            # The first value is the word, the rest are the vector components
            word = values[0]
            try:
                # Try to convert the rest to a numpy array
                vector = np.asarray(values[1:], dtype='float32')
                
                # Check if it's the correct dimension (e.g., 300)
                if vector.shape[0] == 300:
                    embeddings[word] = vector
                # If it's a header line or malformed, we just skip it quietly
            except ValueError:
                print(f"Skipping malformed data on line {line_count} (Word: {word})")
                continue

    print(f"Successfully loaded {len(embeddings)} vectors.")
    return embeddings

def get_hash_01(data_string):
    """Deterministic hash returning a float between 0 and 1."""
    hash_hex = hashlib.sha256(data_string.encode()).hexdigest()
    return int(hash_hex, 16) / (2**256)

def generate_random_vector(seed_string, dimension=300):
    """Generates a consistent unit vector from a seed."""
    seed_int = int(hashlib.sha256(seed_string.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed_int)
    vec = rng.standard_normal(dimension)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def get_embedding(word, v_dict, dim=300):
    """Retrieves vector for word; returns zeros if missing."""
    return v_dict.get(word.lower(), np.zeros(dim))

def get_synonyms_dynamic(word):
    """Fetches synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            syn_word = lemma.name().replace('_', ' ')
            if syn_word.lower() != word.lower():
                synonyms.add(syn_word)
    return list(synonyms)
##
def get_contextual_candidates(text_list, target_index, window_size=10, top_k=5):
    """
    Use BERT to predict words that fit perfectly into the context.
    - text_list: The full list of words in your document.
    - target_index: The exact position (number) of the word we want to replace.
    - window_size: How many words before and after to include for context.
    - top_k: How many candidate words to return.
    """
    #Figure out the start and end boundaries of our Context Window.
    # use max() and min() so we don't accidentally try to grab words 
    # that don't exist (like asking for 10 words before the very first word).
    start_idx = max(0, target_index - window_size)
    end_idx = min(len(text_list), target_index + window_size + 1)
    
    # Extract the actual words for our window
    window_words = text_list[start_idx:end_idx].copy()
    
    # Calculate where our target word is INSIDE this smaller window
    local_target_index = target_index - start_idx
    
    # Swap the actual word with the literal string "[MASK]"
    window_words[local_target_index] = "[MASK]"
    
    # Join the list back into a single string sentence for BERT to read
    masked_sentence = " ".join(window_words)
    
    try:
        # Feed the sentence to BERT. It returns a list of dictionaries.
        predictions = bert_fill_mask(masked_sentence, top_k=top_k)
        
        # BERT returns data that looks like this:
        # [{'score': 0.9, 'token_str': 'fast'}, {'score': 0.05, 'token_str': 'agile'}]
        # We just want to extract the actual words ('token_str').
        candidates = []
        for pred in predictions:
            # We filter out punctuation or weird sub-words BERT sometimes creates
            word = pred['token_str'].strip()
            if word.isalpha(): 
                candidates.append(word)
                
        return candidates
        
    except Exception as e:
        # If the sentence is too weird or BERT fails, return an empty list safely
        return []
##
def simple_tokenize(text):
    """
    Splits text into words and punctuation marks as separate items.
    Example: "Hello, world!" -> ["Hello", ",", "world", "!"]
    """
    # This regex finds words OR sequences of non-space characters (punctuation)
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def unified_detokenizer(tokens):
    """Joins tokens back into a string without 'floating' punctuation."""
    text = " ".join(tokens)
    # This regex removes the space before punctuation marks
    return re.sub(r'\s+([^\w\s])', r'\1', text)

# --- CORE ALGORITHMS ---

def apply_watermark_v2(text_list, K, tau, v_dict):
    """The Encoder: Content-Anchored Resynchronization."""
    watermarked_text = []
    active_anchor = "START_TOKEN"
    offset = 0
    
    stats = {
        "total_tokens": len(text_list),
        "anchors_found": 0,
        "eligible_for_substitution": 0, # Words with known synonyms
        "actually_substituted": 0      # Words flipped to a synonym
    }

    # Identify protected anchors
    is_anchor = [get_hash_01(f"anchor_{K}_{w}") < tau for w in text_list]

    for i, word in enumerate(text_list):
        if is_anchor[i]:
            active_anchor = word
            offset = 0
            watermarked_text.append(word)
            stats["anchors_found"] += 1
            continue
        
        offset += 1
        # Generate the secret direction R
        seed = f"vector_{K}_{active_anchor}_{offset}"
        r = generate_random_vector(seed)
        
        w_vec = get_embedding(word, v_dict)
        if np.all(w_vec == 0): 
            watermarked_text.append(word)
            continue

        score = np.dot(w_vec, r)
        
        if score > 0: # Already Green
            watermarked_text.append(word)
        else: # Red: Try to swap
            #synonyms = get_synonyms_dynamic(word)
            synonyms = get_contextual_candidates(text_list, i, window_size=10, top_k=5)
            if not synonyms:
                watermarked_text.append(word)
                continue

            stats["eligible_for_substitution"] += 1
            best_synonym = word
            max_score = score
            
            for syn in synonyms:
                syn_vec = get_embedding(syn, v_dict)
                if np.all(syn_vec == 0): continue

                syn_score = np.dot(syn_vec, r)
                if syn_score > max_score:
                    max_score = syn_score
                    if word[0].isupper():
                        best_synonym = syn.capitalize()
                    else:
                        best_synonym = syn.lower()

            if best_synonym.lower() != word.lower():
                stats["actually_substituted"] += 1

            watermarked_text.append(best_synonym)
            
    return watermarked_text,stats

# def apply_watermark_v2(text_list, K, tau, v_dict):
#     """The Consistency Encoder: Pairs words to match colors (Red-Red or Green-Green)."""
#     watermarked_text = list(text_list)
#     active_anchor = "START_TOKEN"
#     offset = 0
#     i = 0

#     while i < len(text_list):
#         word = text_list[i]
        
#         # 1. Handle Anchors
#         if get_hash_01(f"anchor_{K}_{word}") < tau:
#             active_anchor = word
#             offset = 0
#             i += 1
#             continue

#         # 2. Identify the next two non-anchor words to form a pair
#         idx1 = i
#         idx2 = -1
        
#         # Find the second word of the pair, skipping any anchors that appear
#         for j in range(i + 1, len(text_list)):
#             if get_hash_01(f"anchor_{K}_{text_list[j]}") < tau:
#                 break # Anchor found; current pair is incomplete
#             idx2 = j
#             break
        
#         if idx2 == -1: # No second word found before end of text or next anchor
#             i += 1
#             continue

#         # 3. Generate vectors and scores for the pair
#         offset1 = offset + 1
#         offset2 = offset + 2 # Assuming contiguous words for simplicity
        
#         r1 = generate_random_vector(f"vector_{K}_{active_anchor}_{offset1}")
#         r2 = generate_random_vector(f"vector_{K}_{active_anchor}_{offset2}")
        
#         v1 = get_embedding(text_list[idx1], v_dict)
#         v2 = get_embedding(text_list[idx2], v_dict)
        
#         s1 = np.dot(v1, r1)
#         s2 = np.dot(v2, r2)

#         # 4. Check if consistency already exists
#         if (s1 * s2) > 0:
#             # Already Red-Red or Green-Green; leave them alone 
#             pass
#         else:
#             # 5. Mismatch! Choose the easiest word to flip 
#             # Heuristic: Which word has better BERT candidates to match the other's color?
#             candidates1 = get_contextual_candidates(text_list, idx1)
#             candidates2 = get_contextual_candidates(text_list, idx2)
            
#             # Attempt to change word 1 to match color of word 2
#             best_syn1 = find_best_match(candidates1, r1, target_sign=np.sign(s2), v_dict=v_dict)
            
#             # Attempt to change word 2 to match color of word 1
#             best_syn2 = find_best_match(candidates2, r2, target_sign=np.sign(s1), v_dict=v_dict)

#             # Simplified decision: prioritize the swap that yields the highest absolute score
#             if best_syn1:
#                 watermarked_text[idx1] = best_syn1
#             elif best_syn2:
#                 watermarked_text[idx2] = best_syn2

#         # Update loop variables
#         i = idx2 + 1
#         offset += 2
            
#     return watermarked_text

# def find_best_match(candidates, r_vec, target_sign, v_dict):
#     """Helper to find a synonym that matches the target color sign."""
#     best_word = None
#     max_magnitude = -1
    
#     for syn in candidates:
#         v_syn = get_embedding(syn, v_dict)
#         score = np.dot(v_syn, r_vec)
#         if np.sign(score) == target_sign:
#             if abs(score) > max_magnitude:
#                 max_magnitude = abs(score)
#                 best_word = syn
#     return best_word

def detect_watermark(text_list, K, tau, v_dict):
    """The Decoder: Resyncs via anchors and calculates Z-score."""
    green_count = 0
    total_checked = 0
    active_anchor = "START_TOKEN"
    offset = 0
    
    for word in text_list:
        h_val = get_hash_01(f"anchor_{K}_{word}")
        
        if h_val < tau:
            active_anchor = word
            offset = 0
            continue
            
        offset += 1
        seed = f"vector_{K}_{active_anchor}_{offset}"
        r = generate_random_vector(seed)
        
        w_vec = get_embedding(word, v_dict)
        if np.all(w_vec == 0): continue # Skip unknown words

        score = np.dot(w_vec, r)
        if score > 0:
            green_count += 1
        
        total_checked += 1
        
    if total_checked == 0: return 0
    
    # Z-score Math
    expected_mu = 0.5 * total_checked
    standard_dev = np.sqrt(total_checked * 0.5 * 0.5)
    z_score = (green_count - expected_mu) / standard_dev
    
    return z_score

# def detect_watermark(text_list, K, tau, v_dict):
#     """Alternative Decoder: Counts color-matched pairs"""
#     match_count, total_pairs = 0, 0
#     active_anchor, offset, pending_score = "START_TOKEN", 0, None

#     for word in text_list:
#         if get_hash_01(f"anchor_{K}_{word}") < tau:
#             active_anchor, offset, pending_score = word, 0, None
#             continue
            
#         offset += 1
#         r = generate_random_vector(f"vector_{K}_{active_anchor}_{offset}")
#         w_vec = get_embedding(word, v_dict)
#         if np.all(w_vec == 0): continue 
#         current_score = np.dot(w_vec, r)

#         if pending_score is None:
#             pending_score = current_score
#         else:
#             if (pending_score * current_score) > 0: # Pair matches 
#                 match_count += 1
#             total_pairs += 1
#             pending_score = None
            
#     if total_pairs == 0: return 0
    
 
#     expected_mu = 0.5 * total_pairs
#     std_dev = np.sqrt(total_pairs * 0.25)
#     return (match_count - expected_mu) / std_dev

 

if __name__ == "__main__":
    download_nltk_deps()

    # Load Vectors
    glove_vectors = load_glove_embeddings(GLOVE_PATH)