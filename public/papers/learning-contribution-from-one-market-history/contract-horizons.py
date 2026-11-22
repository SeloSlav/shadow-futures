"""Certify the Table 3 transfer horizon at the 66% prior (Section 9.5).

Save procurement-calculations.py in the same directory. Run this file with
Python 3 and NumPy. All bounds and decisions use integer or rational arithmetic.
"""
from fractions import Fraction
from math import ceil, floor, lcm
from pathlib import Path
import runpy

import numpy as np


procurement = runpy.run_path(str(Path(__file__).with_name("procurement-calculations.py")))
contract = procurement["TABLE_3"]
badge_prize = procurement["badge_prize"]
deviation_gains = procurement["deviation_gains"]
GRID = 10000
SCALE = 10**7  # Integer units per euro.
N = 2579
nodes = np.arange(GRID + 1, dtype=np.int64)
target = GRID * 33 // 50


class Dispatch:
    def __init__(self, badge, grade):
        self.informative = grade != int(badge == -1)
        high, low = contract.probabilities(grade, badge)
        self.denom = lcm(high.denominator, low.denominator)
        self.transitions = []
        for next_badge, h, l in [(1, high, low), (-1, 1 - high, 1 - low)]:
            h, l = int(h * self.denom), int(l * self.denom)
            weight = nodes * h + (GRID - nodes) * l
            # Bayes posterior in grid units is left + remainder / weight.
            left, remainder = np.divmod(GRID * nodes * h, weight)
            right = np.minimum(left + 1, GRID)
            nearest = np.where(2 * remainder <= weight, left, right)
            self.transitions.append((next_badge, h, l, left, right, weight - remainder, remainder, nearest))

    def bounds(self, previous, gains):
        # The true optimal surplus is convex as the maximum of affine plan
        # payoffs. A chord between upper nodal bounds dominates it off-grid.
        upper = np.zeros_like(nodes)
        lower_h = np.zeros_like(nodes)
        lower_l = np.zeros_like(nodes)
        for b, h, l, left, right, w_left, w_right, nearest in self.transitions:
            u, policy_h, policy_l = previous[b]
            upper += w_left * u[left] + w_right * u[right]
            # Each outcome selects a feasible continuation plan. Only the
            # selected plan uses the rounded posterior; nature uses exact odds.
            lower_h += h * policy_h[nearest]
            lower_l += l * policy_l[nearest]
        lo_h, hi_h, lo_l, hi_l = gains if self.informative else (0, 0, 0, 0)
        upper = ceil_divide(upper, GRID * self.denom)
        upper += ceil_divide(nodes * hi_h + (GRID - nodes) * hi_l, GRID)
        return upper, lower_h // self.denom + lo_h, lower_l // self.denom + lo_l


def ceil_divide(numerator, denominator):
    return -((-numerator) // denominator)


def gain_bounds(remaining, badge):
    # D_k = 35(1 - 5^-k). After 24 powers the gain is enclosed between
    # its values at D_24 and 35, avoiding very large rational powers.
    k = remaining - 1
    prize = badge_prize(min(k, 24), contract)
    endpoints = (prize, prize if k <= 24 else Fraction(35))
    pairs = zip(*(deviation_gains(badge, p, contract) for p in endpoints))
    return tuple(x for pair in pairs for x in (floor(SCALE * min(pair)), ceil(SCALE * max(pair))))


def certify():
    actions = {(b, g): Dispatch(b, g) for b in (-1, 1) for g in (0, 1)}
    # |g| < 6 euros on D in [0,35], checked at both endpoints by affinity.
    assert max(abs(g) for b in (-1, 1) for p in (0, 35)
               for g in deviation_gains(b, p, contract)) < 6
    # Per order allow three rounding units in addition to that payoff bound.
    # The largest numerator is the grid-weighted pair of outcome interpolants.
    max_denom = max(action.denom for action in actions.values())
    assert (6 * SCALE + 3) * N * GRID * max_denom < np.iinfo(np.int64).max
    previous = {b: tuple(np.zeros_like(nodes) for _ in range(3)) for b in (-1, 1)}
    first_positive = []
    last_negative = {}
    for m in range(1, N + 1):
        current = {}
        for b in (-1, 1):
            usual = int(b == -1)
            gains = gain_bounds(m, b)
            silent = actions[b, usual].bounds(previous, gains)
            trial = actions[b, 1 - usual].bounds(previous, gains)
            u, h, l = trial
            # A positive lower bound is the gain from a specific trial-first
            # plan over silence throughout. A negative upper excludes all such plans.
            lower = int((target * h[target] + (GRID - target) * l[target]) // GRID)
            upper = int(u[target])
            assert lower <= upper
            if m <= 5:
                q = Fraction(33, 50)
                baseline = 73 * m + (Fraction(-105, 4) if b == -1 else Fraction(35, 4)) * (1 - Fraction(1, 5)**m)
                exact = procurement["dispatch_value"](m, q, b, True, 1 - usual) - baseline
                assert lower <= SCALE * exact <= upper
            if b == -1 or m <= N - 2:
                assert upper < 0, (m, b, lower, upper)
                last_negative[b] = (m, upper)
            elif m == N - 1:
                assert lower > 0, (m, b, lower, upper)
                first_positive.append((m, b, lower, upper))
            su, sh, sl = silent
            prefer_trial = nodes * h + (GRID - nodes) * l > nodes * sh + (GRID - nodes) * sl
            current[b] = (np.maximum(u, su), np.where(prefer_trial, h, sh), np.where(prefer_trial, l, sl))
        previous = current
    assert len(first_positive) == 1
    print("Last strictly negative upper bounds (remaining orders, units of 1e-7 euro):", last_negative)
    print("First positive bounds (remaining, badge, lower, upper):", first_positive[0])
    print("At 66%, silence is strictly optimal through 2,578 orders starting at B. "
          "With 2,579, start fast; an A win strictly favors an informative fast order next.")


if __name__ == "__main__":
    certify()
