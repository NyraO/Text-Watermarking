import os

from watermarking.versions.v3_tfidf import watermarking_tfidf as wm
from watermarking.common import attack as atk


def _z(result):
    """Pull a printable z-score line out of a detect() result dict."""
    if not result.get("tested_tokens"):
        return f"n/a ({result.get('reason', 'no testable tokens')})"
    flag = "WM" if result["is_watermarked"] else "--"
    return (f"{result['z_score']:6.2f}  [{flag}]  "
            f"green={result['green_fraction']:.2f} "
            f"({result['green_tokens']}/{result['tested_tokens']})")


def run_experiment(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        original_text = f.read()

    # APPLY WATERMARK
    result = wm.watermark(original_text)
    wm_text = result["watermarked_text"]
    families = result["families"]

    # Save watermarked text to results/
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    results_dir = os.path.join(_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_name))[0]
    out_path = os.path.join(results_dir, f"{base}.tfidf.watermarked.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(wm_text)
    print(f"Watermarked text saved to: {out_path}")

    print("--- Document Statistics ---")
    print(f"Main Sentence (idx={result['main_sentence_index']}): {result['main_sentence']!r}")
    print(f"Families discovered: {len(families)}")
    print(f"Substitutions:       {len(result['changes'])}")

    # Detection uses the families recovered from the watermarked text itself.
    def detect(text):
        return wm.detect(text)

    print(f"\n[BASELINE] Watermarked Z-Score: {_z(detect(wm_text))}\n")

    atk_del = atk.deletion(wm_text, strength=3)
    print(f"1. Deletion (strength=3): {_z(detect(atk_del))}")

    atk_del_rand = atk.delete_random(wm_text, strength=50)
    print(f"2. Delete Random (n=50):   {_z(detect(atk_del_rand))}")

    atk_del_port = atk.delete_portion(wm_text, begin=0, end=100)
    print(f"3. Delete Portion (0-100): {_z(detect(atk_del_port))}")

    atk_reo = atk.reorder(wm_text, strength=5, distance=2)
    print(f"4. Reorder (dist=2):       {_z(detect(atk_reo))}")

    atk_reo_rand = atk.reorder_random_max_dist(wm_text, strength=20, max_distance=5)
    print(f"5. Random Reorder:         {_z(detect(atk_reo_rand))}")

    atk_ins = atk.insertion_attack(wm_text, ratio=0.1, words=["filler", "noise"])
    print(f"6. Insertion (10%):        {_z(detect(atk_ins))}")

    atk_noise = atk.insert_noise_attack(wm_text, ratio=0.1)
    print(f"7. Noise/Typo Attack:      {_z(detect(atk_noise))}")

    atk_gen = atk.generative_attack(wm_text, token="[MOD]", n=10)
    print(f"8. Generative Injection:   {_z(detect(atk_gen))}")

    atk_syn = atk.synonym_attack(wm_text, replace_prob=0.3)
    print(f"9. Synonym Attack (30%):   {_z(detect(atk_syn))}")

    # Paraphrasing (LLM Rewrite via local Ollama server)
    try:
        atk_para = atk.paraphrasing_attack(wm_text, style="formal and concise")
        print(f"10. LLM Paraphrasing:      {_z(detect(atk_para))}")
    except Exception as e:
        print(f"10. LLM Paraphrasing SKIPPED: {e}")

    print(f"\n--- BENCHMARK COMPLETE FOR {file_name} ---")


if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    run_experiment(os.path.join(_ROOT, "data", "inputs", "bbc.txt"))
