"""Reproduce the two-state badge calculations in Learning Contribution from One Market History.

Run with Python 3. Table 2 uses the permanent seller premium, called beta in
this script. At a fixed one-unit quality gap it includes the brand advantage.
"""
from math import exp, fsum, log
from itertools import product


def choice(t, badge, gamma, beta, alpha):
    """Probability that seller A wins, given the last winner's badge."""
    eta = beta + badge * (alpha + gamma * log(t))
    if eta >= 0:
        return 1 / (1 + exp(-eta))
    odds = exp(eta)
    return odds / (1 + odds)


def forward(horizon, gamma, beta, alpha):
    """Expected comparison budget and expected curvature from each badge state."""
    prob_plus = 1.0
    budget = info_plus = info_minus = 0.0
    for t in range(1, horizon + 1):
        p = choice(t, 1, gamma, beta, alpha)
        q = choice(t, -1, gamma, beta, alpha)
        budget += prob_plus * min(p, 1 - p) + (1 - prob_plus) * min(q, 1 - q)
        info_plus += prob_plus * p * (1 - p)
        info_minus += (1 - prob_plus) * q * (1 - q)
        prob_plus = prob_plus * p + (1 - prob_plus) * q
    known = info_plus + info_minus
    adjusted = 4 * info_plus * info_minus / known
    return budget, known, adjusted


def likelihood(path, gamma, beta, alpha):
    mass, badge = 1.0, 1
    for t, win in enumerate(path, 1):
        p = choice(t, badge, gamma, beta, alpha)
        mass *= p if win else 1 - p
        badge = 1 if win else -1
    return mass


def exact_risk(horizon, gamma):
    """Equal-prior Bayes error for (0,1) versus (1/2,1/2)."""
    paired = [(likelihood(path, gamma, 0, 1), likelihood(path, gamma, .5, .5))
              for path in product((0, 1), repeat=horizon)]
    for col in zip(*paired):
        assert abs(fsum(col) - 1) < 1e-11
    return .5 * fsum(min(a, b) for a, b in paired)


def check_forward():
    """Independent enumeration for small horizons, including asymmetric quality."""
    for gamma in (0, 1, 2):
        for beta, alpha in ((0, 1), (.5, .5)):
            observed = [0., 0., 0.]
            for path in product((0, 1), repeat=6):
                mass = likelihood(path, gamma, beta, alpha)
                badge = 1
                for t, win in enumerate(path, 1):
                    p = choice(t, badge, gamma, beta, alpha)
                    observed[0] += mass * min(p, 1 - p)
                    observed[1 if badge == 1 else 2] += mass * p * (1 - p)
                    badge = 1 if win else -1
            budget, pos, neg = observed
            expected = (budget, pos + neg, 4 * pos * neg / (pos + neg))
            assert max(abs(x - y) for x, y in zip(expected, forward(6, gamma, beta, alpha))) < 1e-12


def check_opening_badges(horizon=8, premium=1.0):
    """Verify Proposition 1 by enumerating a fair opening and every continuation."""
    challenger = 1 / (1 + exp(premium))
    expected_diagonal = (horizon - 1) * challenger * (1 - challenger)
    expected_budget = .5 + (horizon - 1) * challenger
    for transferable in (False, True):
        masses, diagonals, cross_terms = [], [], []
        for path in product((0, 1), repeat=horizon):
            badge = 2 * path[0] - 1
            mass, budget, diagonal, cross = .5, .5, 0., 0.
            for winner in path[1:]:
                p = choice(1, badge, 0, 0, premium)
                variance = p * (1 - p)
                budget += min(p, 1 - p)
                diagonal += variance
                cross += badge * variance
                mass *= p if winner else 1 - p
                if transferable:
                    badge = 2 * winner - 1
            assert abs(budget - expected_budget) < 1e-12
            masses.append(mass)
            diagonals.append(mass * diagonal)
            cross_terms.append(mass * cross)
        assert abs(fsum(masses) - 1) < 1e-12
        assert abs(fsum(diagonals) - expected_diagonal) < 1e-12
        assert abs(fsum(cross_terms)) < 1e-12


if __name__ == "__main__":
    check_opening_badges()
    check_forward()
    print("gamma     E[B_10000]     I_known     I_adjusted     risk_T=12")
    for gamma in (0, .5, 1, 2):
        values = forward(10_000, gamma, 0, 1)
        print(f"{gamma:5.1f} {values[0]:14.4f} {values[1]:11.4f} {values[2]:14.4f} {exact_risk(12, gamma):13.5f}")
