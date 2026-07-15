from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from src.data.adaptive_attacks import generate_adaptive_cases
from src.defenses.baseline import prompt_baseline
from src.defenses.detector_only import DetectorOnlyDefense
from src.defenses.dual_signed import DefenseConfig, DualSignedDefense
from src.defenses.react import prompt_react
from src.defenses.spotlighting_only import prompt_spotlighting_only
from src.evaluation.judge import error_verdict, judge_response
from src.evaluation.types import (
    DefenseResult,
    EvaluationCase,
    EvaluationConfig,
)
from src.llm.ollama_client import ModelRequest, OllamaClient
from src.metrics.bootstrap import bootstrap_ci
from src.metrics.classification import compute_confusion, compute_precision_recall_f1
from src.utils.io import dump_json, ensure_dir, timestamp
from src.utils.logging_config import configure_logging

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "test_cases.json"
RESULTS_DIR = ROOT_DIR / "data" / "results_v4"

LOGGER = logging.getLogger("evaluation")

def load_dataset(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def load_partition_indices(partition_file: Path, partition_id: str) -> tuple[List[int], dict]:
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found at {partition_file}")
    data = json.loads(partition_file.read_text(encoding="utf-8"))
    partitions = data.get("partitions", [])
    lookup = {item["partition_id"]: item for item in partitions}
    if partition_id not in lookup:
        raise ValueError(
            f"Partition '{partition_id}' not present in {partition_file}"
        )
    partition = lookup[partition_id]
    indices = partition.get("case_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(
            f"Partition '{partition_id}' has no case indices"
        )
    for idx in indices:
        if not isinstance(idx, int):
            raise ValueError(
                f"Partition '{partition_id}' includes non-integer index: {idx!r}"
            )
    metadata = {
        "partition_id": partition_id,
        "partition_file": str(partition_file.resolve()),
        "requested_chunk_size": data.get("requested_chunk_size"),
        "total_partitions": data.get("total_partitions"),
        "total_cases": data.get("total_cases"),
        "start_index": partition.get("start_index"),
        "end_index": partition.get("end_index"),
        "size": partition.get("size", len(indices)),
    }
    return indices, metadata

def select_cases_by_indices(cases: List[dict], indices: List[int]) -> List[dict]:
    selected: List[dict] = []
    total = len(cases)
    for idx in indices:
        if idx < 0 or idx >= total:
            raise IndexError(f"Case index {idx} out of range (0, {total - 1})")
        selected.append(cases[idx])
    return selected

def stratified_selection(cases: List[dict], sample_size: int, seed: int) -> List[dict]:
    categories: dict[str, List[dict]] = {}
    for case in cases:
        categories.setdefault(case["attack_type"], []).append(case)

    if sample_size > len(cases):
        raise ValueError("Sample size exceeds available cases")

    rng = random.Random(seed)
    ordered_categories = sorted(categories.keys())
    per_category = sample_size // len(ordered_categories)
    remainder = sample_size % len(ordered_categories)

    selection: List[dict] = []
    for idx, category in enumerate(ordered_categories):
        bucket = categories[category]
        bucket_copy = bucket[:]
        rng.shuffle(bucket_copy)
        take = per_category + (1 if idx < remainder else 0)
        if len(bucket_copy) < take:
            raise ValueError(
                f"Not enough cases in category '{category}' (needed {take}, got {len(bucket_copy)})"
            )
        selection.extend(bucket_copy[:take])
    return selection

def build_evaluation_cases(base_cases: Iterable[dict]) -> List[EvaluationCase]:
    evaluation_cases: List[EvaluationCase] = []
    for base in base_cases:
        base_id = base["test_id"]
        meta = base.get("metadata", {})
        adv_case = EvaluationCase(
            test_id=f"{base_id}::adv",
            instruction=base["attacker_instruction"],
            user_tool=base["user_tool"],
            attack_type=base["attack_type"],
            category=base["category"],
            is_adversarial=True,
            metadata={
                "base_id": base_id,
                "variant": "adversarial",
                "attacker_instruction": base["attacker_instruction"],
                "benign_instruction": base["benign_instruction"],
                "source_metadata": meta,
            },
        )
        benign_case = EvaluationCase(
            test_id=f"{base_id}::benign",
            instruction=base["benign_instruction"],
            user_tool=base["user_tool"],
            attack_type=base["attack_type"],
            category="benign",
            is_adversarial=False,
            metadata={
                "base_id": base_id,
                "variant": "benign",
                "attacker_instruction": base["attacker_instruction"],
                "benign_instruction": base["benign_instruction"],
                "source_metadata": meta,
            },
        )
        evaluation_cases.extend([adv_case, benign_case])
    return evaluation_cases

def judge_payload(case: EvaluationCase) -> dict:
    if case.is_adversarial:
        return {
            "attacker_instruction": case.metadata["attacker_instruction"],
            "is_benign": False,
        }
    return {
        "attacker_instruction": case.instruction,
        "is_benign": True,
    }

def run_prompt_defense(
    defense_name: str,
    prompt_builder: Callable[[EvaluationCase], str],
    cases: List[EvaluationCase],
    client: OllamaClient,
    model_name: str,
) -> List[DefenseResult]:
    results: List[DefenseResult] = []
    for case in cases:
        prompt = prompt_builder(case)
        try:
            response = client.generate(
                ModelRequest(
                    model=model_name,
                    prompt=prompt,
                )
            )
            verdict = judge_response(judge_payload(case), response)
        except Exception as exc:
            LOGGER.error(
                "%s defense failed for %s: %s",
                defense_name,
                case.test_id,
                exc,
            )
            response = f"ERROR: {exc}"
            verdict = error_verdict(f"Infrastructure error: {exc}")
        results.append(
            DefenseResult(
                test_id=case.test_id,
                is_adversarial=case.is_adversarial,
                defense_name=defense_name,
                prompt=prompt,
                response=response,
                judge_label=verdict.label,
                judge_confidence=verdict.confidence,
                judge_reason=verdict.reason,
            )
        )
    return results

def run_dual_signed_defense(
    cases: List[EvaluationCase],
    dual_defense: DualSignedDefense,
    defense_name: str = "dual_signed",
) -> List[DefenseResult]:
    results: List[DefenseResult] = []
    for case in cases:
        try:
            output = dual_defense.run(case)
            response = output.final_response
            prompt = output.guard_prompt
            if output.fallback_prompt:
                prompt += "\n\n# FALLBACK PROMPT\n" + output.fallback_prompt
            if output.quarantine_prompt:
                prompt += "\n\n# QUARANTINE PROMPT\n" + output.quarantine_prompt
            verdict = judge_response(judge_payload(case), response)
        except Exception as exc:
            LOGGER.error("Dual-signed defense failed for %s: %s", case.test_id, exc)
            response = f"ERROR: {exc}"
            prompt = "<dual-signed-error>"
            verdict = error_verdict(f"Infrastructure error: {exc}")
        results.append(
            DefenseResult(
                test_id=case.test_id,
                is_adversarial=case.is_adversarial,
                defense_name=defense_name,
                prompt=prompt,
                response=response,
                judge_label=verdict.label,
                judge_confidence=verdict.confidence,
                judge_reason=verdict.reason,
            )
        )
    return results

def run_detector_defense(
    cases: List[EvaluationCase],
    detector: DetectorOnlyDefense,
    defense_name: str = "detector_only",
) -> List[DefenseResult]:
    results: List[DefenseResult] = []
    for case in cases:
        try:
            output = detector.run(case)
            response = output.final_response
            prompt = output.prompt
            verdict = judge_response(judge_payload(case), response)
        except Exception as exc:
            LOGGER.error("Detector-only defense failed for %s: %s", case.test_id, exc)
            response = f"ERROR: {exc}"
            prompt = "<detector-only-error>"
            verdict = error_verdict(f"Infrastructure error: {exc}")
        results.append(
            DefenseResult(
                test_id=case.test_id,
                is_adversarial=case.is_adversarial,
                defense_name=defense_name,
                prompt=prompt,
                response=response,
                judge_label=verdict.label,
                judge_confidence=verdict.confidence,
                judge_reason=verdict.reason,
            )
        )
    return results

ABLATION_CONFIGS: dict[str, DefenseConfig] = {
    "dual_signed_full": DefenseConfig(),
    "abl_no_signing": DefenseConfig(enable_signing=False),
    "abl_no_spotlighting": DefenseConfig(enable_spotlighting=False),
    "abl_no_heuristic": DefenseConfig(enable_heuristic=False),
    "abl_no_quarantine": DefenseConfig(enable_quarantine=False),
    "abl_no_fallback": DefenseConfig(enable_fallback=False),
    "abl_spotlight_delimiter": DefenseConfig(spotlight_mode="delimiter"),
    "abl_spotlight_datamarking": DefenseConfig(spotlight_mode="datamarking"),
}

def summarize_defense(
    defense_name: str,
    cases: List[EvaluationCase],
    results: List[DefenseResult],
) -> dict:

    indices_adv = [
        idx
        for idx, case in enumerate(cases)
        if case.is_adversarial and results[idx].judge_label != "error"
    ]
    indices_benign = [
        idx
        for idx, case in enumerate(cases)
        if not case.is_adversarial and results[idx].judge_label != "error"
    ]

    asr_flags = [1 if results[idx].judge_label == "attack_success" else 0 for idx in indices_adv]
    block_flags = [1 - flag for flag in asr_flags]
    benign_flags = [
        1 if results[idx].judge_label == "attack_success" else 0 for idx in indices_benign
    ]

    asr_ci = bootstrap_ci(asr_flags)
    block_ci = bootstrap_ci(block_flags)
    benign_ci = bootstrap_ci(benign_flags)

    judge_labels = [res.judge_label for res in results]
    adversarial_flags = [case.is_adversarial for case in cases]
    confusion = compute_confusion(judge_labels, adversarial_flags)
    report = compute_precision_recall_f1(confusion)

    summary = {
        "defense_name": defense_name,
        "N_total": len(cases),
        "N_adversarial": len(indices_adv),
        "ASR_percent": asr_ci.rate,
        "ASR_95CI": [asr_ci.ci_low, asr_ci.ci_high],
        "BLOCK_percent": block_ci.rate,
        "BLOCK_95CI": [block_ci.ci_low, block_ci.ci_high],
        "BENIGN_success_percent": benign_ci.rate,
        "BENIGN_95CI": [benign_ci.ci_low, benign_ci.ci_high],
        "confusion": confusion.as_dict,
        "precision_attack_success": report.precision,
        "recall_attack_success": report.recall,
        "f1_attack_success": report.f1,
        "accuracy": report.accuracy,
    }
    return summary

def run_evaluation(
    sample_size: int = 50,
    guard_model: str = "qwen2.5:7b",
    quarantine_model: str = "mistral:7b",
    fallback_model: str = "deepseek-coder:6.7b",
    random_seed: int = 20250917,
    *,
    case_indices: Optional[List[int]] = None,
    run_id: Optional[str] = None,
    selection_metadata: Optional[dict] = None,
    include_extra_baselines: bool = False,
    include_ablations: bool = False,
    include_adaptive: bool = False,
) -> dict:
    dataset = load_dataset(DATA_PATH)
    cases_source = dataset["cases"]

    if case_indices is not None:
        selection_mode = "partition"
        selected_cases = select_cases_by_indices(cases_source, case_indices)
    else:
        selection_mode = "stratified"
        selected_cases = stratified_selection(cases_source, sample_size, random_seed)

    effective_sample_size = len(selected_cases)
    evaluation_cases = build_evaluation_cases(selected_cases)

    run_dir_name = run_id or timestamp()
    run_dir = ensure_dir(RESULTS_DIR / run_dir_name)
    configure_logging(run_dir)

    LOGGER.info(
        "Selected %s adversarial cases (mode=%s, total evaluations=%s)",
        effective_sample_size,
        selection_mode,
        len(evaluation_cases),
    )

    selection_payload = {
        "selection_mode": selection_mode,
        "selected_ids": [case["test_id"] for case in selected_cases],
        "dataset_path": str(DATA_PATH.resolve()),
        "effective_sample_size": effective_sample_size,
    }
    if selection_mode == "partition" and case_indices is not None:
        selection_payload["case_indices"] = case_indices
    else:
        selection_payload["sample_size_requested"] = sample_size
        selection_payload["random_seed"] = random_seed
    if selection_metadata:
        selection_payload["partition"] = selection_metadata

    dump_json(selection_payload, run_dir / "selection.json")

    client = OllamaClient()
    config = EvaluationConfig(
        sample_size=effective_sample_size,
        guard_model=guard_model,
        quarantine_model=quarantine_model,
        fallback_model=fallback_model,
        random_seed=random_seed,
    )

    baseline_results = run_prompt_defense(
        "baseline", prompt_baseline, evaluation_cases, client, guard_model
    )
    react_results = run_prompt_defense(
        "react", prompt_react, evaluation_cases, client, guard_model
    )
    dual_defense = DualSignedDefense(
        client=client,
        guard_model=guard_model,
        quarantine_model=quarantine_model,
        fallback_model=fallback_model,
    )
    dual_results = run_dual_signed_defense(evaluation_cases, dual_defense)

    defense_results = {
        "baseline": baseline_results,
        "react": react_results,
        "dual_signed": dual_results,
    }

    if include_extra_baselines:

        defense_results["spotlighting_only"] = run_prompt_defense(
            "spotlighting_only",
            prompt_spotlighting_only,
            evaluation_cases,
            client,
            guard_model,
        )
        detector = DetectorOnlyDefense(client=client, model=guard_model)
        defense_results["detector_only"] = run_detector_defense(
            evaluation_cases, detector
        )

    if include_ablations:

        for name, abl_config in ABLATION_CONFIGS.items():
            abl_defense = DualSignedDefense(
                client=client,
                guard_model=guard_model,
                quarantine_model=quarantine_model,
                fallback_model=fallback_model,
                config=abl_config,
            )
            defense_results[name] = run_dual_signed_defense(
                evaluation_cases, abl_defense, defense_name=name
            )

    adaptive_cases: List[EvaluationCase] = []
    adaptive_results: dict[str, List[DefenseResult]] = {}
    if include_adaptive:

        adaptive_cases = generate_adaptive_cases(evaluation_cases)
        if adaptive_cases:
            adaptive_defense = DualSignedDefense(
                client=client,
                guard_model=guard_model,
                quarantine_model=quarantine_model,
                fallback_model=fallback_model,
            )
            adaptive_results["adaptive_dual_signed"] = run_dual_signed_defense(
                adaptive_cases, adaptive_defense, defense_name="adaptive_dual_signed"
            )

    summaries = {
        name: summarize_defense(name, evaluation_cases, results)
        for name, results in defense_results.items()
    }
    for name, results in adaptive_results.items():
        summaries[name] = summarize_defense(name, adaptive_cases, results)

    summary_payload = {
        "timestamp": run_dir.name,
        "guard_model": guard_model,
        "quarantine_model": quarantine_model,
        "fallback_model": fallback_model,
        "sample_size_adversarial": effective_sample_size,
        "total_evaluations": len(evaluation_cases),
        "selection_mode": selection_mode,
        **summaries,
    }
    if selection_metadata:
        summary_payload["partition"] = selection_metadata

    details_payload = {
        name: [asdict(result) for result in results]
        for name, results in defense_results.items()
    }
    for name, results in adaptive_results.items():
        details_payload[name] = [asdict(result) for result in results]

    final_payload = {
        "summary": summary_payload,
        "selection": selection_payload,
        "details": details_payload,
    }

    dump_json(final_payload, run_dir / "results.json")
    return final_payload

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt injection defense evaluation")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--guard-model", type=str, default="qwen2.5:7b")
    parser.add_argument("--quarantine-model", type=str, default="mistral:7b")
    parser.add_argument("--fallback-model", type=str, default="deepseek-coder:6.7b")
    parser.add_argument("--random-seed", type=int, default=20250917)
    parser.add_argument("--partition-file", type=Path)
    parser.add_argument("--partition-id", type=str)
    parser.add_argument(
        "--output-tag",
        type=str,
        help="Optional suffix appended to result directory name",
    )
    parser.add_argument(
        "--extra-baselines",
        action="store_true",
        help="Also run spotlighting-only and detector-only baselines",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Also run component ablations of the Dual-Signed defense",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Also run worst-case adaptive attacks against the full defense",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if (args.partition_file is None) ^ (args.partition_id is None):
        raise SystemExit("--partition-file and --partition-id must be provided together")

    case_indices: Optional[List[int]] = None
    selection_metadata: Optional[dict] = None
    run_id: Optional[str] = None
    sample_size = args.sample_size

    if args.partition_file and args.partition_id:
        case_indices, selection_metadata = load_partition_indices(
            args.partition_file,
            args.partition_id,
        )
        sample_size = len(case_indices)
        tag = args.output_tag or selection_metadata.get("partition_id")
        base_ts = timestamp()
        run_id = f"{base_ts}__{tag}" if tag else base_ts
    elif args.output_tag:
        run_id = f"{timestamp()}__{args.output_tag}"

    payload = run_evaluation(
        sample_size=sample_size,
        guard_model=args.guard_model,
        quarantine_model=args.quarantine_model,
        fallback_model=args.fallback_model,
        random_seed=args.random_seed,
        case_indices=case_indices,
        run_id=run_id,
        selection_metadata=selection_metadata,
        include_extra_baselines=args.extra_baselines,
        include_ablations=args.ablations,
        include_adaptive=args.adaptive,
    )
    print(json.dumps(payload, indent=2))
