import math
import random
import re
from nltk.corpus import wordnet as wn
from nltk import pos_tag
# from groq import Groq
from random import randrange
from nltk import sent_tokenize
from act_pas_lib.act_pas import *

def tokenizer(text):
    return re.findall(r'\w+|[^\w\s]', text)

def rebuild_text(tokens):
    # This joins tokens but removes the weird spaces before commas/periods
    return re.sub(r'\s+([^\w\s])', r'\1', tokens if isinstance(tokens, str) else " ".join(tokens))

def copy_paste_attack(watermarked_text, excerpt, dilution_rate, position):
    """
    A  watermarked text is inserted into a larger corpus of unwatermarked text, diluting the watermark’s statistical signal
    Insert watermarked text into excerpt at specified position

    """
    wm_tokens = tokenizer(watermarked_text)
    excerpt_tokens = tokenizer(excerpt)

    size_watermarked = len(wm_tokens)
    total_size_needed = math.ceil(size_watermarked / dilution_rate)
    size_excerpt_needed = total_size_needed - size_watermarked

    cropped_tokens = excerpt_tokens[:size_excerpt_needed]

    before_tokens = cropped_tokens[:position]
    after_tokens = cropped_tokens[position:] # Fixed slice logic

    final_tokens = before_tokens + wm_tokens + after_tokens
    return rebuild_text(final_tokens)

def insertion_attack(watermarked_text, ratio, words):
    """
    Words or sentences are randomly added into the text to change the generated watermark
    """

    tokens = tokenizer(watermarked_text)
    original_length = len(tokens)

    num_words_to_insert = math.ceil(original_length * ratio)

    random_positions = [random.randint(0, original_length) for _ in range(num_words_to_insert)]

    # Sort positions in REVERSE order (insert from end to beginning)
    # This way, earlier positions don't shift when we insert
    random_positions.sort(reverse=True)

    # Insert words at each position
    for pos in random_positions:
        word_to_insert = random.choice(words)
        tokens.insert(pos, word_to_insert)

    return rebuild_text(tokens)


def insert_noise_attack(watermarked_text, ratio, punctuations=[",", ":", ";"]):
    tokens = tokenizer(watermarked_text)
    len_watermarked_text = len(tokens)

    # Calculate how many words to modify
    num_words_to_modify = math.ceil(len_watermarked_text * ratio)

    # Generate UNIQUE random positions
    random_positions = random.sample(range(len_watermarked_text),
                                     min(num_words_to_modify, len_watermarked_text))

    # Apply modifications
    for pos in random_positions:
        # Choose modification type randomly
        modification_type = random.randint(1, 3)

        if modification_type == 1:
            # Add punctuation to the word
            punc = random.choice(punctuations)
            tokens[pos] += punc

        elif modification_type == 2:
            # Delete a random letter from the word
            word = tokens[pos]
            if len(word) > 1:  # Only delete if word has more than 1 character
                index_to_delete = random.randrange(0, len(word))
                modified_word = word[:index_to_delete] + word[index_to_delete + 1:]
                tokens[pos] = modified_word

        else:  # modification_type == 3
            # Insert a random vowel into the word
            word = tokens[pos]
            if len(word) > 0:  # Only insert if word is not empty
                insert_index = random.randrange(0, len(word) + 1)  # +1 to allow appending
                vowels = ["a", "e", "i", "o", "u"]
                modified_word = word[:insert_index] + random.choice(vowels) + word[insert_index:]
                tokens[pos] = modified_word

    return rebuild_text(tokens)


def deletion(text, strength):
    # Use the unified tokenizer instead of split()
    tokens = tokenizer(text) 
    output_tokens = []
    
    for i, token in enumerate(tokens):
        # Only keep tokens that don't match the strength skip
        if (i % strength) != 0:
            output_tokens.append(token)
            
    # Use the rebuild_text helper instead of " ".join()
    return rebuild_text(output_tokens)


def delete_random(text, strength):
    """
    Strength is an integer and decides the amount of words deleted from the text
    """

    
    txt_list = tokenizer(text)
    output_words = []

    rand = []
    count = 0
    dele = 0
    i = 0

    # Generate index of words to be deleted
    rand = random.sample(range(len(txt_list)), strength)

    # sorting the generated indexes
    rand.sort()

    # go through all the words in the text
    for word in txt_list:
        # If a word has a specific index, it is deleted
        if count == rand[dele]:
            if dele < strength - 1:
                dele += 1
        else:
            output_words.append(word)
        count += 1
    return rebuild_text(output_words)


def delete_portion(text, begin, end):
    """
    begin and end determine where the deleted part begins and where it ends
    """

    txt_list = text.splitlines()
    output_words = []

    # count used for skipping words (deletion)
    count = 0

    # iterating over every line and word in that line
    for string in txt_list:
        for word in string.split():

            if (begin > count) or (count > end):
                # writing unskipped words in new file
                output_words.append(word)
            count += 1
    return " ".join(output_words)


def generative_attack(text, token, n):
    """
    Insert a specific token every n words
    """
    
    words = tokenizer(text)
    output_words = []
    for i in range(len(words)):
        output_words.append(words[i])
        if i % n == 0:
            output_words.append(token)
    return rebuild_text(output_words)


def paraphrasing_attack(text, style, temperature=0.6, top_p=0.6, size=1):
    """
    Paraphrase a sentence using LLM
    """

    key = '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'
    task = f"""
    You are a paraphrasing engine. Rewrite the user's text while preserving meaning
    Style: {style}.
    Rules:
    - Do not shorten meaning
    - Do not add new ideas
    - Produce natural and fluent English
    """

    client = Groq(api_key=key)
    res = ""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": task},
            {"role": "user", "content": text}
        ],
        temperature=temperature,
        max_tokens=int(sum([1 for c in text if c.isalpha()]) * 0.25 * size),
        top_p=top_p,
        stream=True,
        stop=None
    )
    for chunk in completion:
        res += chunk.choices[0].delta.content or ""
    return res


def synonym_attack(text, replace_prob=0.2, max_replace_ratio=0.1, seed=None):
    """
    Paraphrase "watermarked text" randomly substituting words with synonyms
    """

    if seed is not None:
        random.seed(seed)
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)  # Tokenize with punctuation preserved
    words_only = [t for t in tokens if t.isalpha()]
    pos_tags = pos_tag(words_only)
    token_word_indices = [i for i, t in enumerate(tokens) if t.isalpha()]
    max_replacements = int(len(words_only) * max_replace_ratio)
    # positions of words to substitute with synonym
    candidate_word_positions = list(range(0, len(words_only), len(words_only) // max_replacements))[:max_replacements]
    replaced = 0
    for wpos in candidate_word_positions:
        if replaced >= max_replacements:
            break
        if random.random() > replace_prob:
            continue
        word, penn_pos = pos_tags[wpos]
        wn_pos = penn_to_wn_pos(penn_pos)
        if wn_pos is None:
            continue
        synonym = get_synonym(word, wn_pos)
        if synonym and synonym.lower() != word.lower():
            token_idx = token_word_indices[wpos]
            if word.isupper():
                tokens[token_idx] = synonym.upper()
            else:
                tokens[token_idx] = synonym.lower()
            replaced += 1
    return rebuild_text(tokens)


def penn_to_wn_pos(penn_pos: str):
    """
    Convert Penn Treebank POS tags to WordNet POS tags
    """
    if penn_pos.startswith("N"):
        return wn.NOUN
    if penn_pos.startswith("V"):
        return wn.VERB
    if penn_pos.startswith("J"):
        return wn.ADJ
    if penn_pos.startswith("R"):
        return wn.ADV
    return None


def get_synonym(word, wn_pos):
    synonyms = wn.synsets(word, pos=wn_pos)
    if not synonyms:
        return None
    candidates = []
    for i in synonyms:
        for lemma in i.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                candidates.append(name)
    return random.choice(candidates) if candidates else None


def reorder(text, strength, distance):
    """
    Strength for choosing words
    distance for distance between original position and new position
    """

    # reading original
   
    txt_list = text.splitlines()
    output_words = []

    # count used for skipping words (deletion)
    count = 0

    # dictionary to map specific counts to words
    marked = {}

    # iterating over every line and word in that line
    for string in txt_list:
        for word in string.split():

            # saving chosen words
            if (count % strength) == 0:
                marked[count] = word

            # unchosen words get added to file
            else:
                output_words.append(word)

            # creating speparate list for iteration -> else length changed while iterating
            for c in list(marked.keys()):

                # if enough distance cover -> saved words get added
                if (c + distance) == count:
                    output_words.append(marked[c])
                    marked.pop(c)

            count += 1
    for c in sorted(marked.keys()):
        output_words.append(marked[c])

    return " ".join(output_words)


def reorder_random(text, strength, distance):
    """
    Strength for choosing amount of moved words
    distance for distance between original position and new position
    """

    # reading original
    txt_list = tokenizer(text)

    # creating new file for manipulated text
    
    output_words = []
    count = 0

    # dictionary to map specific counts to words
    marked = {}

    rand = []
    i = 0

    if strength > len(txt_list):
        raise ValueError("strength must be <= number of lines")

    rand = random.sample(range(len(txt_list)), strength)

    # iterating over every line and word in that line
    for word in txt_list:

        # saving chosen words
        if count in rand:
            marked[count] = word

        # unchosen words get added to file
        else:
            output_words.append(word)

        # creating speparate list for iteration -> else length changed while iterating
        for c in list(marked.keys()):

            # if enough distance cover -> saved words get added
            if (c + distance) == count:
                output_words.append(marked[c])
                marked.pop(c)

        count += 1
    

    # adding words that never reached distance
    for c in sorted(marked.keys()):
        output_words.append(marked[c])
    
    return rebuild_text(output_words)


def reorder_random_max_dist(text, strength, max_distance):
    """
    Strength for choosing amount of moved words
    distance for distance between original position and new position
    """

    # reading original
   
    txt_list = text.split()
    output_words = []


    count = 0

    # dictionary to map specific counts to words
    marked = {}

    rand = []
    i = 0

    if strength > len(txt_list):
        raise ValueError("strength must be <= number of lines")

    rand = random.sample(range(len(txt_list)), strength)

    # iterating over every line and word in that line
    for word in txt_list:

        # saving chosen words
        if count in rand:
            marked[count + (randrange(1, max_distance + 1))] = word

        # unchosen words get added to file
        else:
            output_words.append(word)

        # creating sepeparate list for iteration -> else length changed while iterating
        for c in list(marked.keys()):

            # if enough distance cover -> saved words get added
            if c == count:
                output_words.append(marked[c])
                marked.pop(c)

        count += 1

    # adding words that never reached distance
    for c in sorted(marked.keys()):
        output_words.append(marked[c])
    
    return " ".join(output_words)


def syn_transform(text, strength):
    """
    Strength for one of how many sentences used
    """


    txt_list = text

    # using nltk to split text into single sentences
    sentences = sent_tokenize(txt_list)

    # creating file for saving
    output_sentences = []

    # count for amount of sentences changed
    count = 0

    # iterating over sentences
    for s in sentences:
        # changing sentences from active to passive
        if (count % strength) == 0:
            s_out = active_to_passive(s)
        else:
            s_out = s

        # no space before end of sentence
        s_out = re.sub(r'<.!?>', '', s_out)

        # space after end of sentence
        s_out = s_out.strip() + ' '

        output_sentences.append(s_out)

        count += 1

    return " ".join(output_sentences)


def rand_syn_transform(text, strength):
    """
    Strength for one of how many sentences used
    """
    # using nltk to split text into single sentences
    sentences = sent_tokenize(text)

    output_sentences = []

    # count for amount of sentences changed
    count = 0
    trns = 0
    s_out = ""

    rand = random.sample(range(len(sentences)), strength)
    rand.sort()

    # iterating over sentences
    for s in sentences:
        # changing sentences from active to passive
        if trns < strength and count == rand[trns]:
            s_out = active_to_passive(s)
            trns += 1

        else:
            s_out = s

        # no space before end of sentence
        s_out = re.sub(r'\s+([.!?])', r'\1', s_out)

        # space after end of sentence
        s_out = s_out.strip() + ' '

        output_sentences.append(s_out)

        count += 1
    return " ".join(output_sentences)