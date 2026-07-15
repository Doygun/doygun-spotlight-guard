from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EvaluationCase:
    test_id: str
    instruction: str
    user_tool: str
    attack_type: str
    category: str
    is_adversarial: bool
    metadata: Dict

@dataclass
class DefenseResult:
    test_id: str
    is_adversarial: bool
    defense_name: str
    prompt: str
    response: str
    judge_label: str
    judge_confidence: float
    judge_reason: str

@dataclass
class DefenseSummary:
    defense_name: str
    n_total: int
    n_adversarial: int
    asr_percent: float
    asr_ci: tuple[float, float]
    block_percent: float
    block_ci: tuple[float, float]
    benign_pass_percent: float
    benign_ci: tuple[float, float]
    classification_report: Dict

@dataclass
class EvaluationConfig:
    sample_size: int
    guard_model: str
    quarantine_model: str
    fallback_model: Optional[str] = None
    random_seed: int = 20250917
