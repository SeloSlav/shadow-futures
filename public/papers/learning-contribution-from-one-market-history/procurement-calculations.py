"""Reproduce Tables 1 and 3, the Section 4.3 exposure identity and the Section 9 decisions.

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


def model_probabilities(grade, badge):
    """Return A's award probabilities in H and L. Badge +1 is A and -1 is B."""
    low_standard_odds = Fraction(4) if badge == 1 else Fraction(5, 4)
    low_odds = low_standard_odds * Fraction(6, 5) ** grade
    return probability(grade, badge), low_odds / (1 + low_odds)


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


def dispatch_value(orders, prior, badge, transfer, grade):
    """Expected euros over the horizon with the given first grade and optimal continuation."""
    high, low = model_probabilities(grade, badge)
    win = prior * high + (1 - prior) * low
    current = 100 * win - 8 * grade
    after_win = optimal_profit(orders - 1, prior * high / win, 1 if transfer else badge, transfer)
    after_loss = optimal_profit(
        orders - 1, prior * (1 - high) / (1 - win), -1 if transfer else badge, transfer
    )
    return current + win * after_win + (1 - win) * after_loss


@lru_cache(maxsize=None)
def optimal_profit(orders, prior, badge, transfer):
    if orders == 0:
        return Fraction(0)
    return max(dispatch_value(orders, prior, badge, transfer, grade) for grade in (0, 1))


def verify_dispatch():
    # L's standard odds are 5/4 with B featured and 4 with A featured.
    # Its 6/5 quality multiplier leaves the on-policy award probabilities unchanged.
    for badge in (-1, 1):
        low_model = [model_probabilities(grade, badge)[1] for grade in (0, 1)]
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


def verify_badge_comparison():
    prior = Fraction(33, 50)
    orders = 5
    print("Table 3. Badge rule | Standard first | Fast first | Standard minus fast")
    for transfer, rule in [(False, "B keeps badge"), (True, "Latest winner")]:
        standard, fast = [dispatch_value(orders, prior, -1, transfer, grade) for grade in (0, 1)]
        if transfer:
            assert (standard, fast) == (Fraction(170006759, 506250), Fraction(211724, 625))
        else:
            # Section 9.2 measures surplus over always choosing fast, which earns €52 per lot.
            fast_surplus, standard_surplus = first_order_values(orders, prior)
            assert (standard, fast) == (52 * orders + standard_surplus, 52 * orders + fast_surplus)
        gain = standard - fast
        print(f"{rule:14} | €{float(standard):.2f} | €{float(fast):.2f} | €{float(gain):+.2f}")


def verify_transferable_badge_histories():
    """Follow both award outcomes under the strictly optimal policy in Table 3."""
    prior = Fraction(33, 50)
    # Keep distinct histories, even when they lead to the same badge and belief.
    # Each entry holds the badge and the history's likelihood under H and under L.
    histories = [(-1, Fraction(1), Fraction(1))]
    margins = []
    for remaining in range(5, 0, -1):
        next_histories = []
        for badge, likelihood_h, likelihood_l in histories:
            evidence = prior * likelihood_h + (1 - prior) * likelihood_l
            belief = prior * likelihood_h / evidence
            assert belief == prior
            standard, fast = [dispatch_value(remaining, belief, badge, True, g) for g in (0, 1)]
            assert fast != standard
            grade = int(fast > standard)
            assert grade == int(badge == -1)
            margins.append(abs(fast - standard))

            high, low = model_probabilities(grade, badge)
            assert high == low == (Fraction(3, 5) if badge == -1 else Fraction(4, 5))
            next_histories.append((1, likelihood_h * high, likelihood_l * low))
            next_histories.append((-1, likelihood_h * (1 - high), likelihood_l * (1 - low)))
        histories = next_histories

    assert len(margins) == 31 and min(margins) == Fraction(1, 9)
    assert len(histories) == 32
    assert sum(high for _, high, _ in histories) == sum(low for _, _, low in histories) == 1
    for _, high, low in histories:
        assert high == low > 0
        assert prior * high / (prior * high + (1 - prior) * low) == prior
    print("Transferable badge: verified 31 strict dispatch choices and a 66% posterior on all 32 histories.")


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
    verify_badge_comparison()
    verify_transferable_badge_histories()


if __name__ == "__main__":
    main()
