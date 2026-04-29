import watermark as wm  
import attack as atk 

def run_experiment(file_name):
    with open(file_name, 'r') as f:
        original_text = f.read()
    
    v_dict = wm.load_glove_embeddings(wm.GLOVE_PATH)
    tokens = wm.simple_tokenize(original_text)
    
    # APPLY WATERMARK
    wm_tokens , stats = wm.apply_watermark_v2(tokens, wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"--- Document Statistics ---")
    print(f"Total Tokens:    {stats['total_tokens']}")
    print(f"Anchors Found:   {stats['anchors_found']} ({stats['anchors_found']/stats['total_tokens']*100:.1f}%)")
    print(f"Substitutions:   {stats['actually_substituted']} / {stats['eligible_for_substitution']} attempted")
    wm_text = wm.unified_detokenizer(wm_tokens)

    z_clean = wm.detect_watermark(wm.simple_tokenize(wm_text), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"\n[BASELINE] Watermarked Z-Score: {z_clean:.2f}\n")
    
    # Simple Deletion (Every 3rd token)
    atk_del = atk.deletion(wm_text, strength=3)
    z_del = wm.detect_watermark(wm.simple_tokenize(atk_del), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"1. Deletion (strength=3): {z_del:.2f}")

    # Random Deletion (Delete 50 tokens)
    atk_del_rand = atk.delete_random(wm_text, strength=50)
    z_del_rand = wm.detect_watermark(wm.simple_tokenize(atk_del_rand), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"2. Delete Random (n=50):   {z_del_rand:.2f}")

    # Portion Deletion (Delete first 100 tokens)
    atk_del_port = atk.delete_portion(wm_text, begin=0, end=100)
    z_del_port = wm.detect_watermark(wm.simple_tokenize(atk_del_port), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"3. Delete Portion (0-100): {z_del_port:.2f}")
    
    # Sequential Reorder
    atk_reo = atk.reorder(wm_text, strength=5, distance=2)
    z_reo = wm.detect_watermark(wm.simple_tokenize(atk_reo), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"4. Reorder (dist=2):       {z_reo:.2f}")

    # Random Reorder (Max Distance)
    atk_reo_rand = atk.reorder_random_max_dist(wm_text, strength=20, max_distance=5)
    z_reo_rand = wm.detect_watermark(wm.simple_tokenize(atk_reo_rand), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"5. Random Reorder:         {z_reo_rand:.2f}")
    
    # Simple Insertion
    atk_ins = atk.insertion_attack(wm_text, ratio=0.1, words=["filler", "noise"])
    z_ins = wm.detect_watermark(wm.simple_tokenize(atk_ins), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"6. Insertion (10%):        {z_ins:.2f}")

    # Noise Attack (Punctuation/Typo injection)
    atk_noise = atk.insert_noise_attack(wm_text, ratio=0.1)
    z_noise = wm.detect_watermark(wm.simple_tokenize(atk_noise), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"7. Noise/Typo Attack:      {z_noise:.2f}")

    # Generative/Token Injection
    atk_gen = atk.generative_attack(wm_text, token="[MOD]", n=10)
    z_gen = wm.detect_watermark(wm.simple_tokenize(atk_gen), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"8. Generative Injection:   {z_gen:.2f}")
    
    # Synonym Substitution
    atk_syn = atk.synonym_attack(wm_text, replace_prob=0.3)
    z_syn = wm.detect_watermark(wm.simple_tokenize(atk_syn), wm.SECRET_KEY, wm.TAU, v_dict)
    print(f"9. Synonym Attack (30%):   {z_syn:.2f}")

    # 10. Paraphrasing (LLM Rewrite) missing 

    # Active to Passive Transformation
    try:
        atk_pass = atk.syn_transform(wm_text, strength=2)
        z_pass = wm.detect_watermark(wm.simple_tokenize(atk_pass), wm.SECRET_KEY, wm.TAU, v_dict)
        print(f">> Active/Passive Z: {z_pass:.2f}")
    except Exception as e:
        print(f">>10. Active/Passive FAILED: {e} (Skipping...)")

    print(f"\n--- BENCHMARK COMPLETE FOR {file_name} ---")

if __name__ == "__main__":
   
    run_experiment("bbc.txt")