import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/results_v4/12-06-2026-13-55__full_a100/results.json"
d = json.load(open(path, encoding="utf-8"))
s = d["summary"]

print(f"Run: {s.get('timestamp')}  |  adv={s.get('sample_size_adversarial')}  total={s.get('total_evaluations')}")
print(f"Models: guard={s.get('guard_model')} quar={s.get('quarantine_model')} fb={s.get('fallback_model')}")
print()

keys = [k for k in s if isinstance(s[k], dict) and "ASR_percent" in s[k]]
print(f"{'variant':<26} {'ASR%':>6} {'F1':>7} {'prec':>7} {'rec':>7} {'benign%':>8} {'err':>4}")
print("-" * 70)
for k in keys:
    v = s[k]
    err = v.get("confusion", {}).get("excluded_errors", 0)
    print(f"{k:<26} {v['ASR_percent']:>6.1f} {v['f1_attack_success']:>7.3f} "
          f"{v['precision_attack_success']:>7.3f} {v['recall_attack_success']:>7.3f} "
          f"{v['BENIGN_success_percent']:>8.1f} {err:>4}")

for ak in ("adaptive", "adaptive_attacks", "adaptive_results"):
    if ak in s:
        print(f"\n=== {ak} ===")
        print(json.dumps(s[ak], indent=2, ensure_ascii=False)[:2000])
