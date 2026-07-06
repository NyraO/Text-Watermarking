import os

from watermarking.versions.v4_family_anchors import watermark_new as wm
from watermarking.versions.v4_family_anchors.watermark_new import find_main_sentence, sent_tokenize
from watermarking.common import attack as atk


def _z(result):
    """Pull a printable z-score out of a detect() result dict."""
    if not result.get("tested_tokens"):
        return f"n/a ({result.get('reason', 'no testable tokens')})"
    flag = "WM" if result["is_watermarked"] else "--"
    return f"{result['z_score']:6.2f}  [{flag}]  green={result['green_fraction']:.2f} ({result['green_tokens']}/{result['tested_tokens']})"


def run_experiment(file_name):
    with open(file_name, 'r') as f:
        original_text = f.read()

    # Find the main sentence on the ORIGINAL (unwatermarked) text 
    before_idx, before_sent = find_main_sentence(sent_tokenize(original_text))

    # APPLY WATERMARK
    result = wm.watermark(original_text)
    wm_text = result["watermarked_text"]
    anchor_roots = result["anchor_roots"]
    context_families = result["context_families"]

    # Save the watermarked text to results/
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    results_dir = os.path.join(_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_name))[0]
    ext = os.path.splitext(file_name)[1]
    out_path = os.path.join(results_dir, f"{base}.watermarked{ext}")
    with open(out_path, "w") as f:
        f.write(wm_text)
    print(f"Watermarked text saved to: {out_path}")

    # Find the main sentence on the WATERMARKED text 
    after_idx, after_sent = find_main_sentence(sent_tokenize(wm_text))

    print(f"--- Document Statistics ---")
    print(f"Main Sentence Index: {result['main_sentence_index']}")
    print(f"Main Sentence (before): [{before_idx}] {before_sent}")
    print(f"Main Sentence (after):  [{after_idx}] {after_sent}")
    print(f"Anchor Roots:        {len(anchor_roots)} -> {sorted(anchor_roots)}")
    print(f"Context Families:    {len(context_families)}")
    print(f"Substitutions:       {len(result['changes'])}")

 
    def detect(text):
        return wm.detect(text)

    z_clean = detect(wm_text)
    print(f"\n[BASELINE] Watermarked Z-Score: {_z(z_clean)}\n")

    # Simple Deletion (Every 3rd token)
    atk_del = atk.deletion(wm_text, strength=3)
    print(f"1. Deletion (strength=3): {_z(detect(atk_del))}")

    # Random Deletion (Delete 50 tokens)
    atk_del_rand = atk.delete_random(wm_text, strength=50)
    print(f"2. Delete Random (n=50):   {_z(detect(atk_del_rand))}")

    # Portion Deletion (Delete first 100 tokens)
    atk_del_port = atk.delete_portion(wm_text, begin=0, end=100)
    print(f"3. Delete Portion (0-100): {_z(detect(atk_del_port))}")

    # Sequential Reorder
    atk_reo = atk.reorder(wm_text, strength=5, distance=2)
    print(f"4. Reorder (dist=2):       {_z(detect(atk_reo))}")

    # Random Reorder (Max Distance)
    atk_reo_rand = atk.reorder_random_max_dist(wm_text, strength=20, max_distance=5)
    print(f"5. Random Reorder:         {_z(detect(atk_reo_rand))}")

    # Simple Insertion
    atk_ins = atk.insertion_attack(wm_text, ratio=0.1, words=["filler", "noise"])
    print(f"6. Insertion (10%):        {_z(detect(atk_ins))}")

    # Noise Attack (Punctuation/Typo injection)
    atk_noise = atk.insert_noise_attack(wm_text, ratio=0.1)
    print(f"7. Noise/Typo Attack:      {_z(detect(atk_noise))}")

    # Generative/Token Injection
    atk_gen = atk.generative_attack(wm_text, token="[MOD]", n=10)
    print(f"8. Generative Injection:   {_z(detect(atk_gen))}")

    # Synonym Substitution
    atk_syn = atk.synonym_attack(wm_text, replace_prob=0.3)
    print(f"9. Synonym Attack (30%):   {_z(detect(atk_syn))}")


    print(f"\n--- BENCHMARK COMPLETE FOR {file_name} ---")


if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    run_experiment(os.path.join(_ROOT, "data", "inputs", "text.txt"))
