"""Estimate the permanent seller premium and sublinear reinforcement from awards.

Section 5 of Learning Contribution from One Market History. Initial counts are
zero, counts are never discounted, and 1 denotes a win for seller A. Example:

    echo 110010101001 | python full-memory-estimator.py

With a fixed one-unit quality gap the seller premium combines quality and brand.
The legacy output key "quality" is an alias for "seller_premium".
The terminal share anchors a one-dimensional likelihood on a growing prefix.
The estimator establishes consistency; its output is not a confidence interval.
"""
import json
import sys
from math import exp, fsum, log1p


def fit(winners):
    """Fit Theorem 2, using 0 for the slope when the prefix fit is flat."""
    horizon = len(winners)
    if not horizon or any(y not in (0, 1) for y in winners):
        raise ValueError("expected at least one binary award")
    total_a = sum(winners)
    anchor = log1p(total_a) - log1p(horizon - total_a)
    prefix = min(horizon, max(1, int(log1p(horizon) ** 2)))

    count_a = 0
    contrasts = []
    for n, winner in enumerate(winners[:prefix]):
        log_odds = log1p(count_a) - log1p(n - count_a)
        contrasts.append((log_odds - anchor, winner))
        count_a += winner

    def score(slope):
        terms = []
        for contrast, winner in contrasts:
            index = anchor + slope * contrast
            if index >= 0:
                probability = 1 / (1 + exp(-index))
            else:
                odds = exp(index)
                probability = odds / (1 + odds)
            terms.append(contrast * (winner - probability))
        return fsum(terms)

    if score(0) <= 0:
        alpha = 0.0
    elif score(1) >= 0:
        alpha = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(60):
            middle = (low + high) / 2
            if score(middle) > 0:
                low = middle
            else:
                high = middle
        alpha = (low + high) / 2
    premium = (1 - alpha) * anchor
    return {"awards": horizon, "prefix": prefix, "log_odds_anchor": anchor,
            "reinforcement": alpha, "seller_premium": premium, "quality": premium}


if __name__ == "__main__":
    digits = "".join(sys.stdin.read().split())
    if not digits or set(digits) - {"0", "1"}:
        sys.exit("Supply an ordered sequence of 0 and 1 awards on standard input.")
    print(json.dumps(fit([int(digit) for digit in digits]), indent=2))
