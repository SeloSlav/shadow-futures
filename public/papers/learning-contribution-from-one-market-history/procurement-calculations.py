"""Reproduce Table 1 and the binary procurement exposure identity in Section 4.3.

Run with Python 3. All award probabilities are stipulated. Each exposure weight
is the accumulated baseline Bernoulli variance in a quality-by-badge cell.
"""
from fractions import Fraction
from itertools import product


def probability(quality_gap, badge):
    """A has twice B's baseline odds, a 3/2 quality multiplier and a 2x badge factor."""
    odds = 2 * Fraction(3, 2) ** quality_gap * Fraction(2) ** badge
    return odds / (1 + odds)


def quality_exposure(weights):
    """Closed form with entries (grade 0, grade 1) for each badge holder."""
    return sum(w0 * w1 / (w0 + w1) for w0, w1 in weights if w0 + w1)


def projected_residual(weights):
    """Independently fit one nuisance level per badge and sum squared residuals."""
    total = Fraction(0)
    for w0, w1 in weights:
        if w0 + w1:
            mean = w1 / (w0 + w1)
            total += w0 * mean**2 + w1 * (1 - mean)**2
    return total


def main():
    print("Badge holder | Probability at grade 0 | Probability at grade 1 | Net gain")
    for badge, holder, expected in [(-1, "B", Fraction(2)), (1, "A", Fraction(-16, 7))]:
        usual = probability(0, badge)
        faster = probability(1, badge)
        net = 100 * (faster - usual) - 8
        assert net == expected
        odds_ratio = (faster / (1 - faster)) / (usual / (1 - usual))
        assert odds_ratio == Fraction(3, 2)
        print(f"{holder:11} | {str(usual):22} | {str(faster):22} | €{float(net):+.2f}")

    for cells in product(map(Fraction, range(4)), repeat=4):
        weights = (cells[:2], cells[2:])
        diagnostic = quality_exposure(weights)
        assert diagnostic == projected_residual(weights)
        assert (diagnostic > 0) == any(w0 and w1 for w0, w1 in weights)
    print("Verified the within-badge exposure identity for 256 cell-weight combinations.")


if __name__ == "__main__":
    main()
