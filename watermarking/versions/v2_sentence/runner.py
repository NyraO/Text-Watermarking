from watermarking.versions.v2_sentence import sentence_watermark as swm
from watermarking.versions.v2_sentence import attack as atk
# v2's own attack.py has no LLM paraphraser; reuse the shared one
from watermarking.common.attack import paraphrasing_attack
import os

def save_text(filename, text_content):
    """Helper to save text to the repo-level results/ directory."""
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    outdir = os.path.join(_root, "results")
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text_content)
    print(f"  -> Saved: {filepath}")


def run_sentence_experiment(file_name):
    print(f"--- STARTING MACRO-SEMANTIC EXPERIMENT ON {file_name} ---")
    with open(file_name, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    # 1. Apply Sentence Watermark (No GloVe loading required!)
    print("\nEncoding Macro-Semantic Watermark... (This may take a moment for T5 to paraphrase)")
    wm_text, stats = swm.apply_sentence_watermark(original_text, swm.SECRET_KEY)
    
    print("\n" + "="*50)
    print("WATERMARKED TEXT:")
    print("="*50)
    print(wm_text)

    print(f"--- Document Statistics ---")
    print(f"Total Sentences:     {stats['total_sentences']}")
    print(f"Anchors Protected:   {stats['anchors_protected']}")
    print(f"Sentences Rewritten: {stats['filler_substituted']}")
    
    # 2. Baseline Detection (No Attack)
    z_clean = swm.detect_sentence_watermark(wm_text, swm.SECRET_KEY)
    print(f"\n[BASELINE] Clean Watermarked Z-Score: {z_clean:.2f}\n")
    print("--- ATTACK RESULTS ---")
    
    # 1. Simple Deletion (Every 3rd token)
    atk_del = atk.deletion(wm_text, strength=3)
    z_del = swm.detect_sentence_watermark(atk_del, swm.SECRET_KEY)
    print(f"1. Deletion (strength=3):  {z_del:.2f}")

    # 2. Random Deletion (Delete 50 tokens)
    atk_del_rand = atk.delete_random(wm_text, strength=50)
    z_del_rand = swm.detect_sentence_watermark(atk_del_rand, swm.SECRET_KEY)
    print(f"2. Delete Random (n=50):   {z_del_rand:.2f}")

    # 3. Portion Deletion (Delete first 100 tokens)
    atk_del_port = atk.delete_portion(wm_text, begin=0, end=100)
    z_del_port = swm.detect_sentence_watermark(atk_del_port, swm.SECRET_KEY)
    print(f"3. Delete Portion (0-100): {z_del_port:.2f}")
    
    # 4. Sequential Reorder
    atk_reo = atk.reorder(wm_text, strength=5, distance=2)
    z_reo = swm.detect_sentence_watermark(atk_reo, swm.SECRET_KEY)
    print(f"4. Reorder (dist=2):       {z_reo:.2f}")

    # 5. Random Reorder (Max Distance)
    atk_reo_rand = atk.reorder_random_max_dist(wm_text, strength=20, max_distance=5)
    z_reo_rand = swm.detect_sentence_watermark(atk_reo_rand, swm.SECRET_KEY)
    print(f"5. Random Reorder:         {z_reo_rand:.2f}")
    
    # 6. Simple Insertion
    atk_ins = atk.insertion_attack(wm_text, ratio=0.1, words=["filler", "noise"])
    z_ins = swm.detect_sentence_watermark(atk_ins, swm.SECRET_KEY)
    print(f"6. Insertion (10%):        {z_ins:.2f}")

    # 7. Noise Attack (Punctuation/Typo injection)
    atk_noise = atk.insert_noise_attack(wm_text, ratio=0.1)
    z_noise = swm.detect_sentence_watermark(atk_noise, swm.SECRET_KEY)
    print(f"7. Noise/Typo Attack:      {z_noise:.2f}")
    print("="*50)
    print(atk_noise)

    # 8. Generative/Token Injection
    atk_gen = atk.generative_attack(wm_text, token="[MOD]", n=10)
    z_gen = swm.detect_sentence_watermark(atk_gen, swm.SECRET_KEY)
    print(f"8. Generative Injection:   {z_gen:.2f}")
    
    # 9. Synonym Substitution
    try:
        atk_syn = atk.synonym_attack(wm_text, replace_prob=0.3)
        z_syn = swm.detect_sentence_watermark(atk_syn, swm.SECRET_KEY)
        print(f"9. Synonym Attack (30%):   {z_syn:.2f}")
    except Exception as e:
         print(f"9. Synonym FAILED: {e}")

    # 10. Paraphrasing (LLM Rewrite via local Ollama server)
    try:
        atk_para = paraphrasing_attack(wm_text, style="formal and concise")
        z_para = swm.detect_sentence_watermark(atk_para, swm.SECRET_KEY)
        print(f"10. LLM Paraphrasing:      {z_para:.2f}")
    except Exception as e:
        print(f"10. LLM Paraphrasing SKIPPED: {e}")

    # 11. Active to Passive Transformation
    try:
        atk_pass = atk.syn_transform(wm_text, strength=2)
        z_pass = swm.detect_sentence_watermark(atk_pass, swm.SECRET_KEY)
        print(f"11. Active/Passive:        {z_pass:.2f}")
    except Exception as e:
        print(f"11. Active/Passive FAILED: {e}")

    print(f"\n--- BENCHMARK COMPLETE FOR {file_name} ---")

if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    run_sentence_experiment(os.path.join(_ROOT, "data", "inputs", "bbc.txt"))