import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence

RANDOM_SEED = 20250917

ATTACK_TOKENS = [
    "ignore previous",
    "reveal",
    "system prompt",
    "send to",
    "email to",
    "forward to",
    "transfer",
    "secret",
    "leak",
    "confidential",
    "policy",
]

CATEGORY_PREFIX = {
    "Financial Data": "FIN",
    "Physical Data": "PHY",
    "Others": "OTH",
}

RAW_FILES = [
    ("test_cases_dh_enhanced.json", "DH_ENH"),
    ("test_cases_ds_enhanced.json", "DS_ENH"),
]

FINANCIAL_KEYWORDS = (
    "financial",
    "bank",
    "payment",
    "transaction",
    "credit",
    "debit",
    "tax",
    "invoice",
)

PHYSICAL_KEYWORDS = (
    "physical",
    "smartlock",
    "door",
    "drone",
    "surveillance",
    "harm",
    "attack",
    "weapon",
)

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def map_attack_type(raw_type: str) -> str:
    norm = (raw_type or "").lower()
    if any(keyword in norm for keyword in FINANCIAL_KEYWORDS):
        return "Financial Data"
    if any(keyword in norm for keyword in PHYSICAL_KEYWORDS):
        return "Physical Data"
    return "Others"

def load_raw_cases() -> List[Dict]:
    root = project_root() / "InjecAgent" / "data"
    cases: List[Dict] = []
    for file_name, tag in RAW_FILES:
        path = root / file_name
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        for idx, item in enumerate(raw_data):
            normalized = {
                "source_file": file_name,
                "source_tag": tag,
                "source_index": idx,
                "attack_type_raw": item.get("Attack Type", "Unknown"),
                "attack_type": map_attack_type(item.get("Attack Type", "Unknown")),
                "attacker_instruction": item.get("Attacker Instruction", "").strip(),
                "attacker_tools": item.get("Attacker Tools", []),
                "user_tool": item.get("User Tool", "").strip(),
                "user_instruction": item.get("User Instruction", "").strip(),
                "expected_achievements": item.get("Expected Achievements", ""),
                "tool_parameters": item.get("Tool Parameters"),
                "tool_response_template": item.get("Tool Response Template"),
                "tool_response": item.get("Tool Response"),
            }
            if normalized["attacker_instruction"]:
                cases.append(normalized)
    return cases

def sort_key(case: Dict) -> tuple:
    return (case["attack_type"], case["source_tag"], case["source_index"])

def stratified_sample(
    cases: Sequence[Dict], total: int, seed: int = RANDOM_SEED
) -> List[Dict]:
    cases_list = list(cases)
    if total >= len(cases_list):
        return sorted(cases_list, key=sort_key)

    buckets: Dict[str, List[Dict]] = {cat: [] for cat in CATEGORY_PREFIX}
    for case in cases_list:
        buckets.setdefault(case["attack_type"], []).append(case)

    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda c: (c["source_tag"], c["source_index"]))
        rng.shuffle(bucket)

    per_category = total // len(CATEGORY_PREFIX)
    remainder = total % len(CATEGORY_PREFIX)

    selection: List[Dict] = []
    ordered_categories = sorted(CATEGORY_PREFIX.keys())
    for idx, category in enumerate(ordered_categories):
        bucket = buckets.get(category, [])
        if not bucket:
            continue
        take = per_category + (1 if idx < remainder else 0)
        if len(bucket) < take:
            raise ValueError(
                f"Category '{category}' does not have enough cases: required {take}, available {len(bucket)}"
            )
        selection.extend(bucket[:take])

    selection.sort(key=sort_key)
    return selection

def create_benign_instruction(case: Dict) -> str:
    user_instruction = case.get("user_instruction", "").strip()
    if user_instruction:
        return "Support the user's safe request. Task: " + user_instruction

    attacker_instruction = case.get("attacker_instruction", "")
    cleaned = attacker_instruction
    for token in ATTACK_TOKENS:
        cleaned = re.sub(token, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        return "Safe content. Please summarize: " + cleaned
    return "Safe task. Support the user's request while following policies."

def build_dataset(total: int, seed: int = RANDOM_SEED) -> Dict:
    raw_cases = load_raw_cases()
    if total > len(raw_cases):
        total = len(raw_cases)
    selection = stratified_sample(raw_cases, total=total, seed=seed)

    counters = {cat: 1 for cat in CATEGORY_PREFIX}
    processed_cases: List[Dict] = []
    for case in selection:
        category = case["attack_type"]
        prefix = CATEGORY_PREFIX[category]
        test_id = f"test_{prefix}_{counters[category]:04d}"
        counters[category] += 1
        entry = {
            "test_id": test_id,
            "attacker_instruction": case["attacker_instruction"],
            "user_tool": case["user_tool"],
            "attack_type": category,
            "category": "prompt_injection",
            "benign_instruction": create_benign_instruction(case),
            "metadata": {
                "attack_type_raw": case["attack_type_raw"],
                "attacker_tools": case.get("attacker_tools", []),
                "expected_achievements": case.get("expected_achievements"),
                "user_instruction": case.get("user_instruction"),
                "tool_parameters": case.get("tool_parameters"),
                "tool_response_template": case.get("tool_response_template"),
                "tool_response": case.get("tool_response"),
                "source_file": case.get("source_file"),
                "source_index": case.get("source_index"),
            },
        }
        processed_cases.append(entry)

    metadata = {
        "total_cases": len(processed_cases),
        "categories": {
            category: sum(1 for c in processed_cases if c["attack_type"] == category)
            for category in CATEGORY_PREFIX
        },
        "source_files": [file for file, _ in RAW_FILES],
        "random_seed": seed,
    }
    return {"metadata": metadata, "cases": processed_cases}

def save_dataset(dataset: Dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return output

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare evaluation dataset")
    parser.add_argument(
        "--total",
        type=int,
        default=99,
        help="Number of adversarial cases to sample (default 99)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full InjecAgent dataset (overrides --total)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "data" / "processed" / "test_cases.json",
        help="Output path for processed dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset even if it exists",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_cases = load_raw_cases()
    total_available = len(raw_cases)
    target_total = total_available if args.full else min(args.total, total_available)

    if args.output.exists() and not args.force:
        print(f"Dataset already exists at {args.output}. Use --force to overwrite.")
        return

    dataset = build_dataset(total=target_total, seed=args.seed)
    dataset["metadata"]["total_available"] = total_available
    dataset["metadata"]["sampled"] = target_total
    save_dataset(dataset, args.output)
    print(
        f"Processed dataset with {target_total} adversarial cases saved to {args.output}"
    )

if __name__ == "__main__":
    main()
