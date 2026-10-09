"""Synthetic, exact-kernel two-point benchmark; no field data.

Run with the bundled Python interpreter. Output is deterministic for the pinned
seed and NumPy version recorded in paper2_results.json. Classification uses the
exact sequential likelihood ratio, never a fitted surrogate.
"""
from pathlib import Path
import json
import math
import time
import numpy as np

SEED = 20261009
N_PER_LAW = 5000
HORIZONS = (16, 256, 2048, 8192)
MARKET_LENGTH = 16
BETAS = (-0.5, 0.5)
OUT = Path(__file__).resolve().parent


def p1(beta, n1, n0):
    w1 = np.exp(beta) * (1.0 + n1) ** 2
    w0 = (1.0 + n0) ** 2
    return w1 / (w0 + w1)


def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "mc_se": float(np.std(values, ddof=1) / math.sqrt(len(values))),
    }


def run_design(reset, seed):
    rng = np.random.default_rng(seed)
    true_beta = np.repeat(np.array(BETAS), N_PER_LAW)
    n1 = np.zeros(len(true_beta), dtype=np.int64)
    n0 = np.zeros_like(n1)
    llr = np.zeros(len(true_beta))
    budget = np.zeros_like(llr)
    information = np.zeros_like(llr)
    rows = []
    for t in range(max(HORIZONS)):
        if reset and t % MARKET_LENGTH == 0:
            n1.fill(0)
            n0.fill(0)
        pm = p1(BETAS[0], n1, n0)
        pp = p1(BETAS[1], n1, n0)
        pt = p1(true_beta, n1, n0)
        budget += np.minimum(pt, 1.0 - pt)
        information += pt * (1.0 - pt)
        y = rng.random(len(pt)) < pt
        llr += np.where(y, np.log(pp / pm), np.log1p(-pp) - np.log1p(-pm))
        n1 += y
        n0 += ~y
        T = t + 1
        if T not in HORIZONS:
            continue
        ties = np.abs(llr) <= 1e-12
        error = np.where(true_beta < 0, llr > 0, llr < 0).astype(float)
        error[ties] = 0.5
        per_law = [error[i * N_PER_LAW:(i + 1) * N_PER_LAW] for i in range(2)]
        err = float(0.5 * sum(np.mean(x) for x in per_law))
        err_se = float(0.5 * math.sqrt(sum(np.var(x, ddof=1) / N_PER_LAW for x in per_law)))
        if np.count_nonzero(error) == 0:
            # Each law has a 97.5% one-sided exact zero-error bound; their
            # intersection gives a >=95% upper bound on equal-prior risk.
            risk_ci = [0.0, 1.0 - 0.025 ** (1.0 / N_PER_LAW)]
            risk_ci_method = "two exact per-law zero-error bounds, Bonferroni"
        else:
            risk_ci = [max(0.0, err - 1.96 * err_se), min(1.0, err + 1.96 * err_se)]
            risk_ci_method = "balanced per-law normal Monte Carlo interval"
        # Forecast a hypothetical next allocation in the current market before
        # any new reset: single-path age is T, replicated-market age is 16.
        # These are forecasts under two fixed models, not posterior forecasts.
        forecast_timing = "after current market allocations, before any reset; ages T versus 16"
        disagreement = np.abs(p1(BETAS[1], n1, n0) - p1(BETAS[0], n1, n0))
        row = {
            "observations": T,
            "markets": T // MARKET_LENGTH if reset else 1,
            "bayes_error_mc": err,
            "bayes_error_mc_se": err_se,
            "bayes_error_mc_95_interval": risk_ci,
            "interval_method": risk_ci_method,
            "errors_by_law": [float(np.mean(x)) for x in per_law],
            "tie_count": int(np.count_nonzero(ties)),
            "comparison_budget": summarize(budget),
            "conditional_fisher_information": summarize(information),
            "one_step_forecast_disagreement": summarize(disagreement),
            "forecast_timing": forecast_timing,
        }
        assert np.all(information <= budget + 1e-9)
        rows.append(row)
    return rows


def exact_length16_risk():
    """Enumerate all 2**16 histories as an independent validation."""
    n1 = np.zeros(1, dtype=np.int64)
    n0 = np.zeros(1, dtype=np.int64)
    qm = np.ones(1)
    qp = np.ones(1)
    for _ in range(MARKET_LENGTH):
        pm = p1(BETAS[0], n1, n0)
        pp = p1(BETAS[1], n1, n0)
        qm = np.concatenate((qm * (1 - pm), qm * pm))
        qp = np.concatenate((qp * (1 - pp), qp * pp))
        n1 = np.concatenate((n1, n1 + 1))
        n0 = np.concatenate((n0 + 1, n0))
    assert abs(float(np.sum(qm)) - 1.0) < 1e-12
    assert abs(float(np.sum(qp)) - 1.0) < 1e-12
    return float(0.5 * np.sum(np.minimum(qm, qp)))


def main():
    start = time.perf_counter()
    results = {
        "seed": SEED,
        "replicates_per_law": N_PER_LAW,
        "numpy_version": np.__version__,
        "betas": BETAS,
        "equal_prior_weights": [0.5, 0.5],
        "kernel": "p1=exp(beta)*(1+N1)^2/((1+N0)^2+exp(beta)*(1+N1)^2)",
        "single_history": run_design(False, SEED),
        "independent_length16_markets": run_design(True, SEED + 1),
        "exact_length16_bayes_error": exact_length16_risk(),
        "uniform_no_response_control": [
            {"observations": T, "comparison_budget": T / 2,
             "conditional_fisher_information": 0.0,
             "bayes_error_exact": 0.5,
             "one_step_forecast_disagreement": 0.0} for T in HORIZONS],
        "interpretation": "Finite synthetic Monte Carlo is not evidence proving an infinite-horizon theorem.",
    }
    results["runtime_seconds"] = time.perf_counter() - start
    (OUT / "paper2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
