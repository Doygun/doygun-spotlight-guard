import json
import os
from collections import defaultdict

RUN = "data/results_v4/12-06-2026-13-55__full_a100/results.json"
TEST_CASES = "data/processed/test_cases.json"
OUT = "data/analysis_250.json"

LBL_BLOCK = "blocked"
LBL_SUCCESS = "attack_success"
LBL_ERROR = "error"

CATEGORY_NAMES = {"FIN": "Financial Data", "PHY": "Physical Data", "OTH": "Others"}

def base_id(test_id: str) -> str:
    return test_id.split("::")[0]

def cat_of(test_id: str) -> str:

    parts = base_id(test_id).split("_")
    return parts[1] if len(parts) > 1 else "OTH"

def main():
    d = json.load(open(RUN, encoding="utf-8"))
    tc = json.load(open(TEST_CASES, encoding="utf-8"))
    tool_of = {c["test_id"]: c.get("user_tool", "Unknown") for c in tc["cases"]}

    summary = d["summary"]
    details = d["details"]

    variants = [k for k in summary if isinstance(summary[k], dict)
                and "ASR_percent" in summary[k]]

    out = {
        "run": summary.get("timestamp"),
        "sample_size_adversarial": summary.get("sample_size_adversarial"),
        "total_evaluations": summary.get("total_evaluations"),
        "models": {
            "guard": summary.get("guard_model"),
            "quarantine": summary.get("quarantine_model"),
            "fallback": summary.get("fallback_model"),
        },
        "headline": {},
        "by_category": {},
        "tool_asr": {},
        "tool_fp": {},
    }

    for v in variants:
        s = summary[v]
        out["headline"][v] = {
            "ASR_percent": s["ASR_percent"],
            "ASR_95CI": s.get("ASR_95CI"),
            "BLOCK_percent": s.get("BLOCK_percent"),
            "BENIGN_success_percent": s.get("BENIGN_success_percent"),
            "precision": s["precision_attack_success"],
            "recall": s["recall_attack_success"],
            "f1": s["f1_attack_success"],
            "accuracy": s.get("accuracy"),
            "confusion": s.get("confusion"),
        }

    for v in variants:
        recs = details.get(v, [])
        cat_stats = defaultdict(lambda: {"adv": 0, "success": 0, "block": 0})
        tool_success = defaultdict(int)
        tool_adv_total = 0
        tool_fp = defaultdict(int)

        for r in recs:
            tid = r["test_id"]
            label = r.get("judge_label")
            is_adv = r.get("is_adversarial")
            cat = cat_of(tid)
            tool = tool_of.get(base_id(tid), "Unknown")

            if is_adv:
                cat_stats[cat]["adv"] += 1
                tool_adv_total += 1
                if label == LBL_SUCCESS:
                    cat_stats[cat]["success"] += 1
                    tool_success[tool] += 1
                elif label == LBL_BLOCK:
                    cat_stats[cat]["block"] += 1
            else:

                if label == LBL_BLOCK:
                    tool_fp[tool] += 1

        out["by_category"][v] = {}
        for cat, st in cat_stats.items():
            n = st["adv"] or 1
            out["by_category"][v][CATEGORY_NAMES.get(cat, cat)] = {
                "asr": round(100.0 * st["success"] / n, 2),
                "block": round(100.0 * st["block"] / n, 2),
                "n": st["adv"],
            }

        denom = tool_adv_total or 1
        shares = sorted(
            ((t, round(100.0 * c / denom, 1)) for t, c in tool_success.items()),
            key=lambda x: x[1], reverse=True,
        )
        out["tool_asr"][v] = shares[:5]

        fps = sorted(tool_fp.items(), key=lambda x: x[1], reverse=True)
        out["tool_fp"][v] = [(t, c) for t, c in fps[:5] if c > 0]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("WROTE", OUT)

    print("\n=== HEADLINE ===")
    for v in variants:
        h = out["headline"][v]
        print(f"{v:<26} ASR={h['ASR_percent']:>5.1f}  F1={h['f1']:.3f}  "
              f"prec={h['precision']:.3f}  rec={h['recall']:.3f}  benign={h['BENIGN_success_percent']:.1f}")

    print("\n=== BY CATEGORY (key variants) ===")
    for v in ("baseline", "react", "dual_signed_full"):
        print(f"-- {v}")
        for cat, st in out["by_category"].get(v, {}).items():
            print(f"   {cat:<16} ASR={st['asr']:>5.1f} block={st['block']:>5.1f} n={st['n']}")

    print("\n=== TOOL ASR SHARE (key variants) ===")
    for v in ("baseline", "react", "dual_signed_full"):
        print(f"-- {v}: {out['tool_asr'].get(v)}")

    print("\n=== TOOL FP (key variants) ===")
    for v in ("baseline", "react", "dual_signed_full"):
        print(f"-- {v}: {out['tool_fp'].get(v)}")

if __name__ == "__main__":
    main()
