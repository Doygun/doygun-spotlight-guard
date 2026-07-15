from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

@dataclass
class RateWithCI:
    rate: float
    ci_low: float
    ci_high: float

def bootstrap_ci(values: Iterable[int], n_boot: int = 2000, alpha: float = 0.05, seed: int | None = 1234) -> RateWithCI:
    values_list: List[int] = list(values)
    if not values_list:
        return RateWithCI(rate=0.0, ci_low=0.0, ci_high=0.0)

    arr = np.array(values_list, dtype=float)
    rng = np.random.default_rng(seed)

    rate = float(arr.mean()) * 100.0
    boots = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots.append(float(sample.mean()) * 100.0)

    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return RateWithCI(rate=rate, ci_low=lo, ci_high=hi)
