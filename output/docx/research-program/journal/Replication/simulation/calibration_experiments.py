"""Finite-grid Monte Carlo test inversion with nuisance reinforcement.

Independent null calibration and validation draws. Only NumPy is required.
No asymptotic chi-square approximation or fitted-null bootstrap is used.
"""
from pathlib import Path
import json
import math
import time
import numpy as np

BASE = Path(__file__).resolve().parent
BETAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
ALPHAS = (0.5, 1.0, 1.5, 2.0)
GRID = [(b, a) for b in BETAS for a in ALPHAS]
SEEDS = (20261031, 20261032, 20261033)
R_NULL = 999
R_VALIDATION = 1000
HORIZON = 256
LEVEL = 0.05

def likelihoods(beta, alpha, reset, n, seed):
    rng = np.random.default_rng(seed)
    n1, n0 = np.zeros(n), np.zeros(n)
    bgrid = np.array([b for b, a in GRID])[:, None]
    agrid = np.array([a for b, a in GRID])[:, None]
    ll = np.zeros((len(GRID), n))
    for t in range(HORIZON):
        if reset and t % 16 == 0:
            n1.fill(0)
            n0.fill(0)
        ell = np.log1p(n1) - np.log1p(n0)
        pt = 1 / (1 + np.exp(-(beta + alpha * ell)))
        y = rng.random(n) < pt
        eta = bgrid + agrid * ell[None, :]
        ll += y[None, :] * eta - np.logaddexp(0, eta)
        n1 += y
        n0 += ~y
    return ll

def wilson(x, n):
    z = 1.959963984540054
    p = x / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - half), min(1, center + half)]

def rank_pvalues(null_stats, observed):
    # searchsorted(left) includes all calibration ties in the upper tail.
    return (1 + len(null_stats) - np.searchsorted(null_stats, observed, side="left")) / (len(null_stats) + 1)

def main():
    start = time.perf_counter()
    out = {"grid": GRID, "seeds": SEEDS, "null_draws_per_grid_point_per_seed": R_NULL,
           "validation_draws_per_true_point_per_seed": R_VALIDATION, "horizon": HORIZON,
           "nominal_level": LEVEL, "statistic": "2*(max grid log likelihood - null log likelihood)", "cases": []}
    accumulated = {}
    banks = {}
    for reset in (False, True):
        for seed in SEEDS:
            child_seeds = np.random.SeedSequence(seed + (1000 if reset else 0)).spawn(len(GRID) + len(ALPHAS))
            sorted_nulls = []
            for i, (b, a) in enumerate(GRID):
                ll = likelihoods(b, a, reset, R_NULL, child_seeds[i])
                stat = 2 * (ll.max(axis=0) - ll[i])
                sorted_nulls.append(np.sort(stat))
                banks[f"reset{int(reset)}_seed{seed}_null{i}"] = np.sort(stat)
            for j, a in enumerate(ALPHAS):
                ll = likelihoods(0.5, a, reset, R_VALIDATION, child_seeds[len(GRID) + j])
                stats = 2 * (ll.max(axis=0)[None, :] - ll)
                p = np.stack([rank_pvalues(sorted_nulls[i], stats[i]) for i in range(len(GRID))])
                p = p.reshape(len(BETAS), len(ALPHAS), R_VALIDATION)
                procedures = {
                    "correct_known_alpha": p[:, j] > LEVEL,
                    "wrong_fixed_alpha2": p[:, ALPHAS.index(2.0)] > LEVEL,
                    "baseline_nuisance_union": np.max(p[:, [0, 1, 3]], axis=1) > LEVEL,
                    "expanded_nuisance_union": np.max(p, axis=1) > LEVEL,
                }
                beta_index = BETAS.index(0.5)
                for name, accepted in procedures.items():
                    key = (reset, a, name)
                    entry = accumulated.setdefault(key, {"coverage": [], "cardinality": [], "seed_coverage": []})
                    coverage = accepted[beta_index].astype(int)
                    entry["coverage"].append(coverage)
                    entry["cardinality"].append(accepted.sum(axis=0))
                    entry["seed_coverage"].append(float(coverage.mean()))
            print(f"calibration reset={reset} seed={seed} complete", flush=True)
    for (reset, alpha, procedure), entry in accumulated.items():
        coverage = np.concatenate(entry["coverage"])
        cardinality = np.concatenate(entry["cardinality"])
        out["cases"].append({"reset_length": 16 if reset else None, "true_beta": .5, "true_alpha": alpha,
            "procedure": procedure, "coverage": float(coverage.mean()),
            "coverage_mc_se_conditioning_on_calibration_banks": float(math.sqrt(sum(np.var(x, ddof=1) / len(x) for x in entry["coverage"])) / len(entry["coverage"])),
            "coverage_mc_95_wilson_conditioning_on_calibration_banks": wilson(int(coverage.sum()), len(coverage)),
            "coverage_by_seed_bank": entry["seed_coverage"], "mean_grid_set_cardinality": float(cardinality.mean()),
            "coverage_seed_bank_range": [min(entry["seed_coverage"]), max(entry["seed_coverage"])],
            "coverage_mc_se_over_three_seed_banks": float(np.std(entry["seed_coverage"], ddof=1) / math.sqrt(len(SEEDS))),
            "median_grid_set_cardinality": float(np.median(cardinality)),
            "null_class_contains_truth": procedure in ("correct_known_alpha", "expanded_nuisance_union") or
                (procedure == "baseline_nuisance_union" and alpha != 1.5) or (procedure == "wrong_fixed_alpha2" and alpha == 2.0)})
    out["runtime_seconds"] = time.perf_counter() - start
    out["coverage_uncertainty_note"] = "Validation-draw intervals condition on the three generated calibration banks. Across-bank seed rates are also reported; these intervals do not include all finite-bank variability. Rank coverage theorem averages over calibration and observation randomness."
    (BASE / "calibration_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(BASE / "calibration_banks.npz", **banks)
    print(f"calibration complete in {out['runtime_seconds']:.2f}s", flush=True)

if __name__ == "__main__":
    main()
