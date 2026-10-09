"""Independent arithmetic checks for the synthetic Paper 2 benchmark."""
from pathlib import Path
import importlib.util
import json
import numpy as np

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("benchmark", BASE / "paper2_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

# Independent log-odds expression rather than the weight-ratio implementation.
def logistic_probability(beta, n1, n0):
    eta = beta + 2 * np.log1p(n1) - 2 * np.log1p(n0)
    return 1 / (1 + np.exp(-eta))

indices = np.arange(2 ** 16, dtype=np.uint32)
n1 = np.zeros(len(indices), dtype=np.int64)
n0 = np.zeros_like(n1)
log_minus = np.zeros(len(indices))
log_plus = np.zeros(len(indices))
llr = np.zeros(len(indices))
for t in range(16):
    y = ((indices >> t) & 1).astype(bool)
    pm = logistic_probability(-0.5, n1, n0)
    pp = logistic_probability(0.5, n1, n0)
    assert np.allclose(pm, benchmark.p1(-0.5, n1, n0), rtol=1e-14, atol=1e-15)
    assert np.allclose(pp, benchmark.p1(0.5, n1, n0), rtol=1e-14, atol=1e-15)
    minus_step = np.where(y, np.log(pm), np.log1p(-pm))
    plus_step = np.where(y, np.log(pp), np.log1p(-pp))
    log_minus += minus_step
    log_plus += plus_step
    llr += plus_step - minus_step
    n1 += y
    n0 += ~y
assert np.all(n1 + n0 == 16)
assert np.allclose(llr, log_plus - log_minus, rtol=0, atol=1e-13)
qm = np.exp(log_minus)
qp = np.exp(log_plus)
assert abs(float(qm.sum()) - 1) < 1e-12
assert abs(float(qp.sum()) - 1) < 1e-12
risk = float(0.5 * np.minimum(qm, qp).sum())
assert abs(risk - benchmark.exact_length16_risk()) < 1e-12

# Reset timing: counts reset before observations 1,17,...,8177; the final
# current-market counts at every reported horizon therefore total sixteen.
counts = 0
market_number = 0
timing = []
for t in range(8192):
    if t % 16 == 0:
        counts = 0
        market_number += 1
    counts += 1
    if t + 1 in benchmark.HORIZONS:
        assert counts == 16
        assert market_number == (t + 1) // 16
        timing.append({"observations": t + 1, "markets": market_number, "final_market_age": counts})

results = json.loads((BASE / "paper2_results.json").read_text(encoding="utf-8"))
assert abs(results["exact_length16_bayes_error"] - risk) < 1e-12
for design in ("single_history", "independent_length16_markets"):
    assert [row["observations"] for row in results[design]] == [16, 256, 2048, 8192]
    assert all(row["tie_count"] == 0 for row in results[design])
    for row in results[design]:
        risk_from_laws = 0.5 * sum(row["errors_by_law"])
        assert abs(row["bayes_error_mc"] - risk_from_laws) < 1e-14
        if row["bayes_error_mc"] == 0:
            expected_upper = 1 - 0.025 ** (1 / 5000)
            assert abs(row["bayes_error_mc_95_interval"][1] - expected_upper) < 1e-14

validation = {
    "status": "pass",
    "independent_log_odds_probability_check": "all 65,536 length-16 histories",
    "exact_length16_risk": risk,
    "likelihood_ratio_sign": "positive selects beta=+0.5; negative selects beta=-0.5",
    "reset_horizon_arithmetic": timing,
    "zero_error_upper_risk_percent": 100 * (1 - 0.025 ** (1 / 5000)),
    "zero_error_coverage_scope": "95% simultaneous over both parameter laws at each specified horizon",
    "no_ties_in_recorded_monte_carlo": True,
    "stored_results_unchanged": True,
}
(BASE / "paper2_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
print(json.dumps(validation, indent=2))
