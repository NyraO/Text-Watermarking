import hashlib
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- CONFIGURATION ---
SECRET_KEY = "macro_semantic_key_2026"
# all-MiniLM-L6-v2 outputs 384-dimensional vectors
VECTOR_DIMENSION = 384 

print("Loading Sentence-BERT for sentence embeddings...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading T5 model for sentence paraphrasing...")
# Load tokenizer and model directly instead of using pipeline
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

def download_nltk_deps():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# --- CRYPTO & MATH HELPERS ---

def generate_random_vector(seed_string, dimension=VECTOR_DIMENSION):
    """Generates a consistent unit vector from a seed."""
    seed_int = int(hashlib.sha256(seed_string.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed_int)
    vec = rng.standard_normal(dimension)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def get_sentence_embedding(sentence):
    """Returns the 384D Sentence-BERT embedding."""
    return sbert_model.encode(sentence)

# --- THE MACRO-SEMANTIC LOGIC ---

def is_semantic_anchor(sentence, is_first, is_last):
    """
    Determines if a sentence is part of the 'Semantic Core'.
    
    IDEAL IMPLEMENTATION: 
    Pass the whole text to an LLM and ask: "Return exactly the 3 sentences 
    that represent the Call to Action, Main Takeaway, and Core Purpose."
    
    LOCAL RUNNABLE IMPLEMENTATION:
    We use a heuristic. The first sentence (Intro), the last sentence (Takeaway), 
    and any sentence containing strong discourse markers are locked as anchors.
    """
    if is_first or is_last:
        return True
        
    discourse_markers = [
        "therefore", "crucial", "must", "conclude", "essential", 
        "takeaway", "in summary", "require", "call to action"
    ]
    
    sentence_lower = sentence.lower()
    if any(marker in sentence_lower for marker in discourse_markers):
        return True
        
    return False

def generate_sentence_candidates(sentence, top_k=5):
    """Generates paraphrased versions of a sentence using direct model generation."""
    text = "paraphrase: " + sentence + " </s>"
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    try:
        # Generate the outputs
        outputs = paraphrase_model.generate(
            **inputs, 
            max_length=256, 
            num_return_sequences=top_k, 
            num_beams=top_k, 
            early_stopping=True
        )
        
        # Decode the generated tokens back into text
        candidates = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        
        # Ensure the original sentence is always an option if paraphrases fail
        candidates.append(sentence) 
        # Remove duplicates
        return list(set(candidates))
        
    except Exception as e:
        print(f"Paraphrasing failed for: {sentence}. Error: {e}")
        return [sentence]

# --- CORE ALGORITHMS ---

def apply_sentence_watermark(text, K):
    """The Encoder: Protects core meaning, paraphrases filler sentences."""
    sentences = nltk.sent_tokenize(text)
    watermarked_sentences = []
    
    active_anchor = "START_DOCUMENT"
    offset = 0
    
    stats = {
        "total_sentences": len(sentences),
        "anchors_protected": 0,
        "filler_substituted": 0
    }

    for i, sentence in enumerate(sentences):
        is_first = (i == 0)
        is_last = (i == len(sentences) - 1)
        
        # 1. Identify if this is a load-bearing "Core" sentence
        if is_semantic_anchor(sentence, is_first, is_last):
            active_anchor = sentence # The literal string becomes the anchor hash base
            offset = 0
            watermarked_sentences.append(sentence) # LEAVE UNTOUCHED
            stats["anchors_protected"] += 1
            continue
            
        # 2. It is a filler sentence. Process for watermarking.
        offset += 1
        
        # Generate secret direction based on the last protected core sentence
        seed = f"vector_{K}_{active_anchor}_{offset}"
        r = generate_random_vector(seed)
        
        s_vec = get_sentence_embedding(sentence)
        score = np.dot(s_vec, r)
        
        if score > 0:
            # Already naturally aligned with the watermark (Green)
            watermarked_sentences.append(sentence)
        else:
            # Misaligned (Red). We must paraphrase it to flip the sign.
            candidates = generate_sentence_candidates(sentence)
            
            best_candidate = sentence
            max_score = score
            
            for candidate in candidates:
                cand_vec = get_sentence_embedding(candidate)
                cand_score = np.dot(cand_vec, r)
                
                # We want the candidate that pushes furthest into the Green
                if cand_score > max_score:
                    max_score = cand_score
                    best_candidate = candidate
                    
            if best_candidate != sentence:
                stats["filler_substituted"] += 1
                
            watermarked_sentences.append(best_candidate)
            
    return " ".join(watermarked_sentences), stats


def detect_sentence_watermark(text, K):
    """The Decoder: Finds the semantic anchors and checks the filler vectors."""
    sentences = nltk.sent_tokenize(text)
    
    green_count = 0
    total_checked = 0
    active_anchor = "START_DOCUMENT"
    offset = 0
    
    for i, sentence in enumerate(sentences):
        is_first = (i == 0)
        is_last = (i == len(sentences) - 1)
        
        # The detector MUST agree with the encoder on what an anchor is
        if is_semantic_anchor(sentence, is_first, is_last):
            active_anchor = sentence
            offset = 0
            continue
            
        offset += 1
        seed = f"vector_{K}_{active_anchor}_{offset}"
        r = generate_random_vector(seed)
        
        s_vec = get_sentence_embedding(sentence)
        score = np.dot(s_vec, r)
        
        if score > 0:
            green_count += 1
            
        total_checked += 1
        
    if total_checked == 0: 
        return 0.0 # No filler sentences found to check
        
    # Z-score Calculation
    expected_mu = 0.5 * total_checked
    standard_dev = np.sqrt(total_checked * 0.25) # 0.5 * 0.5
    z_score = (green_count - expected_mu) / standard_dev
    
    return z_score

# --- EXECUTION ---
if __name__ == "__main__":
    download_nltk_deps()
    
    sample_text = """
The invention of rockets is linked inextricably with the invention of 'black powder'. Most
historians of technology credit the Chinese with its discovery. They base their belief on studies of
Chinese writings or on the notebooks of early Europeans who settled in or made long visits to
China to study its history and civilization. It is probable that, some time in the tenth century,
black powder was first compounded from its basic ingredients of saltpetre, charcoal and sulphur.
But this does not mean that it was immediately used to propel rockets. By the thirteenth century,
powder-propelled fire arrows had become rather common. The Chinese relied on this type of
technological development to produce incendiary projectiles of many sorts, explosive grenades
and possibly cannons to repel their enemies. One such weapon was the 'basket of fire' or, as
directly translated from Chinese, the 'arrows like flying leopards'. The 0.7 metre-long arrows,
each with a long tube of gunpowder attached near the point of each arrow, could be fired from a
long, octagonal-shaped basket at the same time and had a range of 400 paces. Another weapon
was the 'arrow as a flying sabre', which could be fired from crossbows. The rocket, placed in a
similar position to other rocket-propelled arrows, was designed to increase the range. A small
iron weight was attached to the 1.5m bamboo shaft, just below the feathers, to increase the
arrow's stability by movin g the centre of gravity to a position below the rocket.
    """
    
    print("\n--- Original Text ---")
    print(sample_text.strip())
    
    print("\n--- Embedding Watermark... ---")
    watermarked_text, stats = apply_sentence_watermark(sample_text, SECRET_KEY)
    
    print("\n--- Watermarked Text ---")
    print(watermarked_text)
    print("\nStats:", stats)
    
    print("\n--- Detecting Watermark ---")
    z_score = detect_sentence_watermark(watermarked_text, SECRET_KEY)
    print(f"Detection Z-Score: {z_score:.2f}")
    
    # Sanity check: Detect on unwatermarked text
    z_score_original = detect_sentence_watermark(sample_text, SECRET_KEY)
    print(f"Original Text Z-Score (Should be ~0): {z_score_original:.2f}")