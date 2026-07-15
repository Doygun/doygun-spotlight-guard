from src.metrics.bootstrap import bootstrap_ci

def test_bootstrap_constant():
    values = [1, 1, 1, 1]
    rate = bootstrap_ci(values, n_boot=100, seed=42)
    assert rate.rate == 100.0
    assert rate.ci_low == 100.0
    assert rate.ci_high == 100.0

def test_bootstrap_bounds():
    values = [0, 1, 0, 1]
    rate = bootstrap_ci(values, n_boot=100, seed=42)
    assert 0.0 <= rate.ci_low <= rate.rate <= rate.ci_high <= 100.0
