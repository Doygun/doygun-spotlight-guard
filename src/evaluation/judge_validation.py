from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

LABELS = ("attack_success", "blocked")

def _load_details(input_path: Path) -> Dict[str, List[dict]]:
    with input_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    details = data.get("details")
    if not isinstance(details, dict):
        raise ValueError("Expected a 'details' mapping in the results file.")
    return details

def export_sample(
    input_path: Path,
    output_path: Path,
    per_defense: int,
    seed: int,
) -> int:
    details = _load_details(input_path)
    rng = random.Random(seed)
    rows: List[dict] = []

    for defense_name, records in details.items():
        adversarial = [r for r in records if r.get("is_adversarial")]
        benign = [r for r in records if not r.get("is_adversarial")]
        half = max(1, per_defense // 2)
        chosen: List[dict] = []
        for bucket in (adversarial, benign):
            if not bucket:
                continue
            k = min(half, len(bucket))
            chosen.extend(rng.sample(bucket, k))
        for record in chosen:
            rows.append(
                {
                    "test_id": record.get("test_id", ""),
                    "defense_name": defense_name,
                    "is_adversarial": record.get("is_adversarial", ""),
                    "judge_label": record.get("judge_label", ""),
                    "human_label": "",
                    "response": (record.get("response", "") or "").replace("\n", " "),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test_id",
                "defense_name",
                "is_adversarial",
                "judge_label",
                "human_label",
                "response",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)

def cohen_kappa(pairs: List[Tuple[str, str]]) -> float:

    n = len(pairs)
    if n == 0:
        return float("nan")

    observed = sum(1 for a, b in pairs if a == b) / n

    expected = 0.0
    for label in LABELS:
        p_human = sum(1 for a, _ in pairs if a == label) / n
        p_judge = sum(1 for _, b in pairs if b == label) / n
        expected += p_human * p_judge

    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)

def score_sample(input_path: Path) -> dict:
    pairs: List[Tuple[str, str]] = []
    skipped = 0
    with input_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            human = (row.get("human_label") or "").strip().lower()
            judge = (row.get("judge_label") or "").strip().lower()
            if human not in LABELS or judge not in LABELS:
                skipped += 1
                continue
            pairs.append((human, judge))

    n = len(pairs)
    raw_agreement = (
        sum(1 for a, b in pairs if a == b) / n if n else float("nan")
    )
    kappa = cohen_kappa(pairs)

    table = {
        h: {j: sum(1 for a, b in pairs if a == h and b == j) for j in LABELS}
        for h in LABELS
    }

    return {
        "n_scored": n,
        "n_skipped": skipped,
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "contingency_human_rows_judge_cols": table,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Judge vs. human validation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="Export a sample for manual labeling")
    p_export.add_argument("--input", type=Path, required=True)
    p_export.add_argument("--output", type=Path, required=True)
    p_export.add_argument("--per-defense", type=int, default=30)
    p_export.add_argument("--seed", type=int, default=20250917)

    p_score = sub.add_parser("score", help="Score human vs. judge agreement")
    p_score.add_argument("--input", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "export":
        count = export_sample(args.input, args.output, args.per_defense, args.seed)
        print(f"Wrote {count} rows to {args.output}")
        print("Fill the 'human_label' column with 'attack_success' or 'blocked'.")
    elif args.command == "score":
        report = score_sample(args.input)
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
