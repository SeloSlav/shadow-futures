"""Finite-horizon synthetic experiments for Measuring Comparison Opportunities.

Only NumPy and Python's standard library are required. No field observations.
Running this file writes recorded Monte Carlo and exact-enumeration results.
"""
from pathlib import Path
import json
import math
import time
import numpy as np

BASE = Path(__file__).resolve().parent
HORIZONS = (16, 256, 2048, 8192)
SEEDS = (20261021, 20261022, 20261023)
N_PER_LAW = 2000

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def probabilities(beta, alpha, n1, n0):
    return logistic(beta + alpha * (np.log1p(n1) - np.log1p(n0)))

def mc_mean(x):
    return {"mean": float(np.mean(x)), "mc_se": float(np.std(x, ddof=1) / math.sqrt(len(x)))}

def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]

def simulate_pair(alpha, separation, reset_length=None, retention=0.0, response=False, seed=SEEDS[0], n=N_PER_LAW):
    rng = np.random.default_rng(seed)
    response_rng = np.random.default_rng(seed + 900000)
    beta = np.repeat([-separation, separation], n)
    size = len(beta)
    n1 = np.zeros(size)
    n0 = np.zeros(size)
    llr = np.zeros(size)
    wrong_reset_llr = np.zeros(size)
    fresh_n1 = np.zeros(size)
    fresh_n0 = np.zeros(size)
    budget = np.zeros(size)
    information = np.zeros(size)
    iba = np.zeros(size)
    iaa = np.zeros(size)
    rows = []
    qm, qp = logistic(-separation), logistic(separation)
    qt = logistic(beta)
    for t in range(max(HORIZONS)):
        if reset_length is not None and t % reset_length == 0:
            n1 *= retention
            n0 *= retention
            fresh_n1.fill(0)
            fresh_n0.fill(0)
        ell = np.log1p(n1) - np.log1p(n0)
        pm = logistic(-separation + alpha * ell)
        pp = logistic(separation + alpha * ell)
        pt = logistic(beta + alpha * ell)
        budget += np.minimum(pt, 1 - pt)
        i = pt * (1 - pt)
        information += i
        iba += i * ell
        iaa += i * ell * ell
        y = rng.random(size) < pt
        llr += np.where(y, np.log(pp) - np.log(pm), np.log1p(-pp) - np.log1p(-pm))
        if reset_length is not None:
            fm = probabilities(-separation, alpha, fresh_n1, fresh_n0)
            fp = probabilities(separation, alpha, fresh_n1, fresh_n0)
            wrong_reset_llr += np.where(y, np.log(fp) - np.log(fm), np.log1p(-fp) - np.log1p(-fm))
            fresh_n1 += y
            fresh_n0 += ~y
        n1 += y
        n0 += ~y
        if response:
            z = response_rng.random(size) < qt
            llr += np.where(z, np.log(qp) - np.log(qm), np.log1p(-qp) - np.log1p(-qm))
            information += qt * (1 - qt)
        T = t + 1
        if T not in HORIZONS:
            continue
        def errors(ratio):
            e = np.where(beta < 0, ratio > 0, ratio < 0).astype(float)
            e[np.abs(ratio) <= 1e-12] = 0.5
            return e
        row = {"observations": T, "error": errors(llr), "budget": budget.copy(),
               "information": information.copy(), "information_beta_alpha": iba.copy(),
               "information_alpha_alpha": iaa.copy(),
               "forecast_gap": np.abs(probabilities(separation, alpha, n1, n0) - probabilities(-separation, alpha, n1, n0)),
               "wrong_reset_error": errors(wrong_reset_llr) if reset_length is not None else None}
        if not response:
            assert np.all(information <= budget + 1e-9)
        rows.append(row)
    return rows

def pool_rows(seed_rows, config):
    rows = []
    for j, T in enumerate(HORIZONS):
        # Fixed equal-prior design: concatenate each truth separately.
        def pool(field):
            return np.concatenate([np.concatenate([s[j][field][:N_PER_LAW] for s in seed_rows]),
                                   np.concatenate([s[j][field][N_PER_LAW:] for s in seed_rows])])
        error = pool("error")
        per_law_n = N_PER_LAW * len(seed_rows)
        laws = [error[:per_law_n], error[per_law_n:]]
        risk = float(0.5 * sum(x.mean() for x in laws))
        se = float(0.5 * math.sqrt(sum(np.var(x, ddof=1) / per_law_n for x in laws)))
        if np.count_nonzero(error) == 0:
            ci = [0.0, 1 - 0.025 ** (1 / per_law_n)]
            method = "97.5% exact one-sided per-law zero-error bounds; Bonferroni"
        else:
            ci = [max(0, risk - 1.959963984540054 * se), min(1, risk + 1.959963984540054 * se)]
            method = "balanced per-law normal Monte Carlo interval"
        ib = pool("information")
        iba = pool("information_beta_alpha")
        iaa = pool("information_alpha_alpha")
        # Schur complement of estimated expected Fisher matrix, separately by truth.
        schur = []
        for sl in (slice(0, per_law_n), slice(per_law_n, None)):
            a, b, c = float(ib[sl].mean()), float(iba[sl].mean()), float(iaa[sl].mean())
            schur.append(a - b * b / c if c > 0 else a)
        row = {"observations": T, "bayes_risk": risk, "risk_mc_se": se, "risk_mc_95_interval": ci,
               "risk_interval_method": method, "errors_by_law": [float(x.mean()) for x in laws],
               "risk_by_seed": [float(s[j]["error"].mean()) for s in seed_rows],
               "budget": mc_mean(pool("budget")), "information_beta_beta": mc_mean(ib),
               "information_beta_alpha": mc_mean(iba), "information_alpha_alpha": mc_mean(iaa),
               "efficient_information_beta_unknown_alpha_by_law": schur,
               "forecast_gap": mc_mean(pool("forecast_gap"))}
        if seed_rows[0][j]["wrong_reset_error"] is not None:
            wrong = pool("wrong_reset_error")
            row["risk_assuming_fresh_resets"] = mc_mean(wrong)
        rows.append(row)
    return {**config, "rows": rows}

def exact_short_market(alpha, separation, length):
    n1, n0 = np.zeros(1), np.zeros(1)
    qm, qp = np.ones(1), np.ones(1)
    bm = bp = im = ip = 0.0
    for _ in range(length):
        pm = probabilities(-separation, alpha, n1, n0)
        pp = probabilities(separation, alpha, n1, n0)
        bm += float(np.dot(qm, np.minimum(pm, 1 - pm)))
        bp += float(np.dot(qp, np.minimum(pp, 1 - pp)))
        im += float(np.dot(qm, pm * (1 - pm)))
        ip += float(np.dot(qp, pp * (1 - pp)))
        qm = np.concatenate((qm * (1 - pm), qm * pm))
        qp = np.concatenate((qp * (1 - pp), qp * pp))
        n1 = np.concatenate((n1, n1 + 1))
        n0 = np.concatenate((n0 + 1, n0))
    assert abs(float(qm.sum()) - 1) < 1e-12
    assert abs(float(qp.sum()) - 1) < 1e-12
    return {"alpha": alpha, "separation": separation, "length": length,
            "exact_bayes_risk": float(0.5 * np.minimum(qm, qp).sum()),
            "affinity": float(np.sqrt(qm * qp).sum()),
            "expected_budget": 0.5 * (bm + bp), "expected_information": 0.5 * (im + ip)}

def main():
    start = time.perf_counter()
    out = {"numpy_version": np.__version__, "seeds": SEEDS, "replicates_per_law_per_seed": N_PER_LAW,
           "horizons": HORIZONS, "response_channel_rng": "Independent NumPy default_rng(seed + 900000); recipient stream unchanged", "core": [], "stress": [], "exact_short_markets": [], "design_choices": []}
    for alpha in (0.5, 1.0, 2.0):
        for d in (0.25, 0.5, 1.0):
            for reset in (None, 16):
                config = {"alpha": alpha, "separation": d, "reset_length": reset, "retention": 0.0, "response": False}
                runs = [simulate_pair(alpha, d, reset, seed=s) for s in SEEDS]
                out["core"].append(pool_rows(runs, config))
                print(f"core alpha={alpha} beta=+/-{d} reset={reset} finished", flush=True)
    stress_configs = [
        {"alpha": 2.0, "separation": 0.5, "reset_length": None, "retention": 0.0, "response": True},
        *[{"alpha": 2.0, "separation": 0.5, "reset_length": 16, "retention": r, "response": False} for r in (0.5, 0.9, 1.0)],
    ]
    for config in stress_configs:
        runs = [simulate_pair(**config, seed=s) for s in SEEDS]
        out["stress"].append(pool_rows(runs, config))
        print(f"stress {config} finished", flush=True)
    # Retention=1 reproduces a single inherited history exactly with the same seeds.
    single = next(c for c in out["core"] if c["alpha"] == 2 and c["separation"] == .5 and c["reset_length"] is None)
    unchanged = next(c for c in out["stress"] if c["retention"] == 1)
    assert all(a["bayes_risk"] == b["bayes_risk"] and a["budget"] == b["budget"] for a, b in zip(single["rows"], unchanged["rows"]))
    for alpha in (.5, 1., 2.):
        for d in (.25, .5, 1.):
            for length in (1, 2, 4, 8, 12, 16):
                out["exact_short_markets"].append(exact_short_market(alpha, d, length))
    for alpha in (.5, 1., 2.):
        for d in (.25, .5, 1.):
            candidates = [x for x in out["exact_short_markets"] if x["alpha"] == alpha and x["separation"] == d]
            for startup in (0, 4, 16, 64):
                best = max(candidates, key=lambda x: -math.log(x["affinity"]) / (x["length"] + startup))
                out["design_choices"].append({"alpha": alpha, "separation": d, "startup_cost": startup,
                    "chosen_length": best["length"], "affinity_exponent_per_cost": -math.log(best["affinity"]) / (best["length"] + startup)})
    out["runtime_seconds"] = time.perf_counter() - start
    (BASE / "comparison_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"all experiments complete in {out['runtime_seconds']:.2f}s", flush=True)

if __name__ == "__main__":
    main()
