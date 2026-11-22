"""Reproduce Table 1, the Section 4.3 exposure identity and the Section 9 decisions.

Run with Python 3. All award probabilities are stipulated. Each exposure weight
is the accumulated baseline Bernoulli variance in a quality-by-badge cell.
"""
from fractions import Fraction
from functools import lru_cache
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


def standard_outcomes(prior):
    """Chance of each standard-delivery award outcome and posterior chance of H."""
    for high, low in [(Fraction(1, 2), Fraction(5, 9)), (Fraction(1, 2), Fraction(4, 9))]:
        chance = prior * high + (1 - prior) * low
        yield chance, prior * high / chance


def fast_gain(prior):
    """Expected extra euros from fast delivery on a B-badged lot."""
    return 2 * prior - Fraction(32, 9) * (1 - prior)


def pilot_information(prior):
    """Per later lot, value of choosing a grade after observing one standard award."""
    after = sum(p * max(0, fast_gain(q)) for p, q in standard_outcomes(prior))
    return after - max(0, fast_gain(prior))


def first_order_values(orders, prior):
    """Expected surplus over always choosing fast, for each possible first action."""
    fast = sequential_surplus(orders - 1, prior)
    standard = -fast_gain(prior) + sum(
        p * sequential_surplus(orders - 1, q) for p, q in standard_outcomes(prior)
    )
    return fast, standard


@lru_cache(maxsize=None)
def sequential_surplus(orders, prior):
    if orders == 0:
        return Fraction(0)
    return max(first_order_values(orders, prior))


def verify_dispatch():
    # L's standard odds are 5/4 with B featured and 4 with A featured.
    # Its 6/5 quality multiplier leaves the on-policy award probabilities unchanged.
    for badge, usual_odds in [(-1, Fraction(5, 4)), (1, Fraction(4))]:
        low_model = []
        for grade in (0, 1):
            odds = usual_odds * Fraction(6, 5) ** grade
            low_model.append(odds / (1 + odds))
        policy_grade = int(badge == -1)
        assert low_model[policy_grade] == probability(policy_grade, badge)
        low_gain = 100 * (low_model[1] - low_model[0]) - 8
        assert low_gain == (Fraction(-32, 9) if badge == -1 else Fraction(-152, 29))

    prior = Fraction(33, 50)
    gain = fast_gain(prior)
    information = pilot_information(prior)
    assert gain == Fraction(1, 9) and information == Fraction(47, 4050)
    assert [q for _, q in standard_outcomes(prior)] == [Fraction(297, 467), Fraction(297, 433)]
    assert gain / information == Fraction(450, 47)
    assert 9 * information < gain < 10 * information
    print(f"One pilot: current cost €{float(gain):.4f}, value per later lot €{float(information):.5f}.")

    # Independently enumerate all four pilot-contingent batch policies in euros.
    for q in [Fraction(0), Fraction(16, 25), prior, Fraction(160, 241), Fraction(2, 3), Fraction(1)]:
        for later in (0, 4, 9, 10):
            best_without = 52 + later * max(52, 52 - fast_gain(q))
            plans = []
            for after_win, after_loss in product(("fast", "standard"), repeat=2):
                value = 0
                for weight, win_chance in [(q, Fraction(1, 2)), (1 - q, Fraction(5, 9))]:
                    profit = {"fast": 52, "standard": 100 * win_chance}
                    future = win_chance * profit[after_win] + (1 - win_chance) * profit[after_loss]
                    value += weight * (100 * win_chance + later * future)
                plans.append(value)
            assert max(plans) - best_without == later * pilot_information(q) - fast_gain(q)
        if q >= Fraction(16, 25):
            assert pilot_information(q) == max(0, (160 - 241 * q) / 81)

    # Work backward with both actions available at every date.
    for orders in range(1, 6):
        fast, standard = first_order_values(orders, prior)
        assert (standard > fast) == (orders >= 5)
        print(f"{orders - 1} later lots: standard minus fast today, €{float(standard - fast):+.5f}.")
    assert first_order_values(5, prior)[1] == Fraction(912517, 23619600)


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
    verify_dispatch()


if __name__ == "__main__":
    main()
