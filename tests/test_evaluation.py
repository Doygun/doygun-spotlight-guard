import json

import pytest

from src.evaluation.optimized_evaluation2 import (
    build_evaluation_cases,
    load_partition_indices,
    select_cases_by_indices,
    stratified_selection,
)

def _make_case(idx: int, category: str, attack_type: str) -> dict:
    return {
        "test_id": f"test_{idx:03d}",
        "attacker_instruction": f"attack {idx}",
        "user_tool": "tool.call",
        "attack_type": attack_type,
        "category": category,
        "benign_instruction": f"benign {idx}",
        "metadata": {},
    }

def test_stratified_selection_balanced():
    categories = [
        ("prompt_injection", "Financial Data"),
        ("prompt_injection", "Physical Data"),
        ("prompt_injection", "Others"),
    ]
    dataset = []
    counter = 0
    for category, attack_type in categories:
        for _ in range(5):
            dataset.append(_make_case(counter, category, attack_type))
            counter += 1
    sample = stratified_selection(dataset, sample_size=6, seed=42)
    assert len(sample) == 6
    counts = {}
    for case in sample:
        counts[case["attack_type"]] = counts.get(case["attack_type"], 0) + 1
    assert all(count >= 2 for count in counts.values())

def test_build_evaluation_cases_pairs():
    base_cases = [_make_case(idx, "prompt_injection", "Financial Data") for idx in range(3)]
    evaluation_cases = build_evaluation_cases(base_cases)
    assert len(evaluation_cases) == 6
    assert sum(1 for case in evaluation_cases if case.is_adversarial) == 3
    assert sum(1 for case in evaluation_cases if not case.is_adversarial) == 3

def test_select_cases_by_indices_order():
    cases = [_make_case(idx, "prompt_injection", "Financial Data") for idx in range(5)]
    selected = select_cases_by_indices(cases, [4, 0, 2])
    assert [case["test_id"] for case in selected] == ["test_004", "test_000", "test_002"]

def test_load_partition_indices(tmp_path):
    partition_path = tmp_path / "partitions.json"
    data = {
        "requested_chunk_size": 2,
        "total_cases": 4,
        "total_partitions": 2,
        "partitions": [
            {
                "partition_id": "part_01",
                "start_index": 0,
                "end_index": 1,
                "size": 2,
                "case_indices": [0, 1],
            }
        ],
    }
    partition_path.write_text(json.dumps(data), encoding="utf-8")
    indices, metadata = load_partition_indices(partition_path, "part_01")
    assert indices == [0, 1]
    assert metadata["partition_id"] == "part_01"
    assert metadata["size"] == 2
    assert metadata["partition_file"].endswith("partitions.json")
