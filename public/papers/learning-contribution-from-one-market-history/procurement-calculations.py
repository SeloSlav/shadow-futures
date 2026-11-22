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


def first_grade_gap(orders, prior, badge, transfer):
    """Expected total profit with standard first minus profit with fast first."""
    standard = dispatch_value(orders, prior, badge, transfer, 0)
    fast = dispatch_value(orders, prior, badge, transfer, 1)
    return standard - fast


@lru_cache(maxsize=None)
def prior_partition(orders, badge, transfer):
    """Partition [0, 1] where both first-grade values and their maximum are affine."""
    boundaries = {Fraction(0), Fraction(1)}
    if orders == 0:
        return tuple(sorted(boundaries))

    for grade in (0, 1):
        high, low = model_probabilities(grade, badge)
        for prob_h, prob_l, winner in [(high, low, 1), (1 - high, 1 - low, -1)]:
            next_badge = winner if transfer else badge
            for posterior in prior_partition(orders - 1, next_badge, transfer):
                # Invert Bayes' rule to find the prior that reaches this boundary.
                prior = posterior * prob_l / ((1 - posterior) * prob_h + posterior * prob_l)
                boundaries.add(prior)

    # Between these boundaries each outcome probability times its continuation
    # value is affine. A strict sign change therefore has one exact linear root.
    knots = sorted(boundaries)
    gaps = [first_grade_gap(orders, q, badge, transfer) for q in knots]
    for left, right, g_left, g_right in zip(knots, knots[1:], gaps, gaps[1:]):
        if g_left * g_right < 0:
            root = left - (right - left) * g_left / (g_right - g_left)
            boundaries.add(root)
    return tuple(sorted(boundaries))


def verify_prior_range():
    # Every gap is affine on each partition interval. Its signs at all the
    # boundaries certify the ordering everywhere, including zero and one.
    fixed_cutoff = Fraction(4065536, 6132251)
    transfer_cutoff = Fraction(5366848, 19978195)
    rules = [(False, fixed_cutoff, 472392, "Fixed"), (True, transfer_cutoff, 3280500, "Transferable")]
    for transfer, expected_cutoff, scale, label in rules:
        knots = prior_partition(5, -1, transfer)
        gaps = {q: first_grade_gap(5, q, -1, transfer) for q in knots}
        ties = [q for q, gap in gaps.items() if gap == 0]
        assert knots[0] == 0 and knots[-1] == 1 and ties == [expected_cutoff]
        cutoff = ties[0]
        assert all(gap * (cutoff - q) > 0 for q, gap in gaps.items() if q != cutoff)
        # Verify the two adjacent pieces against the local euro formula in Section 9.3.
        i = knots.index(cutoff)
        for q in (knots[i-1], knots[i+1]):
            assert gaps[q] == (cutoff.numerator - cutoff.denominator * q) / scale
        print(f"{label} badge: a unique tie on the first order at {cutoff} ({100 * float(cutoff):.6f}%).")
    assert transfer_cutoff < Fraction(16, 25) < Fraction(33, 50) < fixed_cutoff
    assert fast_gain(Fraction(16, 25)) == 0
    print("Strict reversal (standard first with a fixed badge, fast first under transfer): "
          f"{transfer_cutoff} < prior < {fixed_cutoff}.")
    return fixed_cutoff


def verify_five_order_no_learning_range():
    """Certify optimal uninformative dispatch at every reachable state and prior."""
    # At an interior prior an award leaves beliefs unchanged iff the models agree
    # on its win probability. All outcomes have positive probability in both models.
    silent_grade = {}
    for badge in (-1, 1):
        for grade in (0, 1):
            high, low = model_probabilities(grade, badge)
            assert 0 < high < 1 and 0 < low < 1
            if high == low:
                assert badge not in silent_grade
                silent_grade[badge] = grade
    assert silent_grade == {-1: 1, 1: 0}

    # Four rival wins reach a final B-badged order. Its fast-dispatch gain is
    # affine, strictly increasing and negative below this exact break-even prior.
    low_gain, high_gain = fast_gain(Fraction(0)), fast_gain(Fraction(1))
    assert low_gain < 0 < high_gain
    cutoff = -low_gain / (high_gain - low_gain)
    assert cutoff == Fraction(16, 25)

    # While beliefs are unchanged, the same prior applies at every state. Test
    # all affine pieces of the Bellman action gap, including ties, with optimal
    # continuation even after an informative deviation.
    for orders in range(1, 6):
        badges = (-1,) if orders == 5 else (-1, 1)
        for badge in badges:
            knots = prior_partition(orders, badge, True)
            priors = [cutoff] + [q for q in knots if q > cutoff]
            for q in priors:
                gap = first_grade_gap(orders, q, badge, True)
                advantage = gap if silent_grade[badge] == 0 else -gap
                if orders == 1 and badge == -1 and q == cutoff:
                    assert advantage == 0
                else:
                    assert advantage > 0

    for q in (Fraction(0), cutoff, Fraction(1)):
        assert -first_grade_gap(1, q, -1, True) == fast_gain(q)
    assert all(posterior != cutoff for _, posterior in standard_outcomes(cutoff))
    print("Transferable badge, initially uncertain vendor: all optimal policies preserve beliefs "
          f"on all five orders for {cutoff} < prior < 1. At {cutoff}, only the final B-badged "
          "order ties; fast preserves beliefs, standard updates them. Priors 0 and 1 stay fixed.")
    return cutoff


def verify_trial_elimination(no_learning_cutoff, fixed_cutoff):
    """Intersect the strict fixed-badge trial region with five uninformative awards under transfer."""
    # The transfer check above certifies strict nonlearning for all interior
    # priors above its cutoff. Standard with B featured is an informative trial.
    high, low = model_probabilities(0, -1)
    assert high != low
    knots = sorted(set(prior_partition(5, -1, False)) | {no_learning_cutoff, fixed_cutoff})
    # Between consecutive knots the fixed action gap has constant sign.
    midpoints = [(left + right) / 2 for left, right in zip(knots, knots[1:])]
    for q in knots + midpoints:
        trial_required = first_grade_gap(5, q, -1, False) > 0
        no_learning_required = no_learning_cutoff < q < 1
        assert (trial_required and no_learning_required) == (no_learning_cutoff < q < fixed_cutoff)

    # Above the upper cutoff fast stays strictly optimal at every remaining
    # horizon under the fixed badge. Fast leaves the prior unchanged.
    for orders in range(1, 6):
        knots = prior_partition(orders, -1, False)
        for q in [fixed_cutoff] + [k for k in knots if k > fixed_cutoff]:
            gap = first_grade_gap(orders, q, -1, False)
            if orders == 5 and q == fixed_cutoff:
                assert gap == 0
            else:
                assert gap < 0
    width_pp = 100 * (fixed_cutoff - no_learning_cutoff)
    assert width_pp == Fraction(14089536, 6132251)
    print(f"Transfer eliminates a strictly preferred fixed-badge trial on every history for "
          f"{no_learning_cutoff} < prior < {fixed_cutoff}: {float(width_pp):.2f} percentage points.")


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
    fixed_cutoff = verify_prior_range()
    no_learning_cutoff = verify_five_order_no_learning_range()
    verify_trial_elimination(no_learning_cutoff, fixed_cutoff)


if __name__ == "__main__":
    main()
