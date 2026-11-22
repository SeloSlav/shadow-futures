"""One full-memory sales history. Run with Python 3 and NumPy from this directory."""
import json
from pathlib import Path

import numpy as np


HORIZONS = (256, 2048, 8192)
GRID = [(b, a) for b in (.25, .5, .75) for a in (.25, .5, .75)]
PAIR = [GRID.index((.5, .5)), GRID.index((.25, .75))]
CALIBRATION, VALIDATION = 999, 2000


def simulate(truth_ids, seed):
    """Each column of stored likelihoods is one candidate on the common path."""
    rng = np.random.default_rng(seed)
    beta, alpha = np.asarray(GRID).T
    count = np.zeros(len(truth_ids), dtype=int)
    ll = np.zeros((len(truth_ids), len(GRID)))
    records = {}
    for n in range(max(HORIZONS)):
        states, inv = np.unique(count, return_inverse=True)
        z = np.log1p(states) - np.log1p(n - states)
        eta = beta + z[:, None] * alpha
        win = rng.random(len(count)) < 1 / (1 + np.exp(-eta[inv, truth_ids]))
        ll -= np.logaddexp(0, eta)[inv]
        ll += win[:, None] * eta[inv]
        count += win
        if n + 1 in HORIZONS:
            records[n + 1] = ll.copy()
    return records


def run():
    nulls = np.repeat(np.arange(len(GRID)), CALIBRATION)
    truths = np.repeat(PAIR, VALIDATION)
    bank = simulate(nulls, 2026112201)
    validation = simulate(truths, 2026112202)
    beta = np.array([p[0] for p in GRID])
    rows = []
    for horizon in HORIZONS:
        l0, lv = bank[horizon], validation[horizon]
        dc = 2 * (l0.max(axis=1)[:, None] - l0)
        dv = 2 * (lv.max(axis=1)[:, None] - lv)
        keep = np.zeros_like(dv, dtype=bool)
        for j in range(len(GRID)):
            ordered = np.sort(dc[nulls == j, j])
            ge = CALIBRATION - np.searchsorted(ordered, dv[:, j], side="left")
            keep[:, j] = (1 + ge) / (1 + CALIBRATION) > .05
        assert keep.any(axis=1).all()  # the highest-likelihood candidate has discrepancy zero
        selected = np.stack([keep[:, beta == b].any(axis=1) for b in np.unique(beta)], axis=1)
        retained_width = np.max(np.where(keep, beta, -np.inf), axis=1) - np.min(np.where(keep, beta, np.inf), axis=1)
        for j in PAIR:
            subset = truths == j
            diff = lv[subset, j] - lv[subset, PAIR[1 - PAIR.index(j)]]
            risk = (diff < 0).astype(float) + .5 * (diff == 0)
            rows.append(dict(horizon=horizon,beta=GRID[j][0],alpha=GRID[j][1],
                             wrong=float(risk.mean()),mcse=float(risk.std(ddof=1)/np.sqrt(len(risk))),
                             coverage=float(keep[subset][:, beta == beta[j]].any(axis=1).mean()),
                             mean_grid_span=float(retained_width[subset].mean()),
                             mean_quality_values=float(selected[subset].sum(axis=1).mean())))
    return dict(grid=GRID, calibration_per_cell=CALIBRATION, validation_per_truth=VALIDATION,
                seeds=[2026112201,2026112202], rows=rows)


if __name__ == "__main__":
    result = run()
    Path(__file__).with_name("same_share_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for row in result["rows"]:
        print(row)
