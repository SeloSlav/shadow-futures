"""Reproduce Tables 1 and 3, the Section 4.3 exposure identity and the Section 9 contract results.

Run with Python 3. All award probabilities are stipulated. Each exposure weight
is the accumulated baseline Bernoulli variance in a quality-by-badge cell.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
from typing import Callable, NamedTuple, Tuple


def probability(quality_gap, badge):
    """A has twice B's baseline odds, a 3/2 quality multiplier and a 2x badge factor."""
    odds = 2 * Fraction(3, 2) ** quality_gap * Fraction(2) ** badge
    return odds / (1 + odds)


def model_probabilities(grade, badge):
    """Return A's award probabilities in H and L. Badge +1 is A and -1 is B."""
    low_standard_odds = Fraction(4) if badge == 1 else Fraction(5, 4)
    low_odds = low_standard_odds * Fraction(6, 5) ** grade
    return probability(grade, badge), low_odds / (1 + low_odds)


class Contract(NamedTuple):
    margin: int
    fast_cost: int
    probabilities: Callable[[int, int], Tuple[Fraction, Fraction]]  # H, L


TABLE_3 = Contract(100, 8, model_probabilities)


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


def dispatch_value(orders, prior, badge, transfer, grade, contract=TABLE_3):
    """Expected euros over the horizon with the given first grade and optimal continuation."""
    high, low = contract.probabilities(grade, badge)
    win = prior * high + (1 - prior) * low
    current = contract.margin * win - contract.fast_cost * grade
    after_win = optimal_profit(orders - 1, prior * high / win, 1 if transfer else badge, transfer, contract)
    after_loss = optimal_profit(
        orders - 1, prior * (1 - high) / (1 - win), -1 if transfer else badge, transfer, contract
    )
    return current + win * after_win + (1 - win) * after_loss


@lru_cache(maxsize=None)
def optimal_profit(orders, prior, badge, transfer, contract=TABLE_3):
    if orders == 0:
        return Fraction(0)
    return max(dispatch_value(orders, prior, badge, transfer, grade, contract) for grade in (0, 1))


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


def first_grade_gap(orders, prior, badge, transfer, contract=TABLE_3):
    """Expected total profit with standard first minus profit with fast first."""
    standard = dispatch_value(orders, prior, badge, transfer, 0, contract)
    fast = dispatch_value(orders, prior, badge, transfer, 1, contract)
    return standard - fast


@lru_cache(maxsize=None)
def prior_partition(orders, badge, transfer, contract=TABLE_3):
    """Partition [0, 1] where both first-grade values and their maximum are affine."""
    boundaries = {Fraction(0), Fraction(1)}
    if orders == 0:
        return tuple(sorted(boundaries))

    for grade in (0, 1):
        high, low = contract.probabilities(grade, badge)
        for prob_h, prob_l, winner in [(high, low, 1), (1 - high, 1 - low, -1)]:
            next_badge = winner if transfer else badge
            for posterior in prior_partition(orders - 1, next_badge, transfer, contract):
                # Invert Bayes' rule to find the prior that reaches this boundary.
                prior = posterior * prob_l / ((1 - posterior) * prob_h + posterior * prob_l)
                boundaries.add(prior)

    # Between these boundaries each outcome probability times its continuation
    # value is affine. A strict sign change therefore has one exact linear root.
    knots = sorted(boundaries)
    gaps = [first_grade_gap(orders, q, badge, transfer, contract) for q in knots]
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


def badge_prize(orders, contract):
    """Value of holding the badge under fast with B featured and standard with A featured."""
    s_a, _ = contract.probabilities(0, 1)
    s_b, _ = contract.probabilities(1, -1)
    reward_gap = contract.margin * (s_a - s_b) + contract.fast_cost
    return reward_gap * (1 - (s_a - s_b) ** orders) / (1 - s_a + s_b)


def deviation_gains(badge, prize, contract):
    usual = int(badge == -1)
    s, other = contract.probabilities(usual, badge)
    assert s == other
    costs_saved = (2 * usual - 1) * contract.fast_cost
    return tuple((p - s) * (contract.margin + prize) + costs_saved
                 for p in contract.probabilities(1 - usual, badge))


def first_trial_bound(orders, badge, prior, contract):
    """Bound all trial-first plans by the posterior range i awards can reach."""
    ratios = []
    for b, grade in product((-1, 1), (0, 1)):
        h, l = contract.probabilities(grade, b)
        ratios.extend((h / l, (1 - h) / (1 - l)))
    low, high = min(ratios), max(ratios)

    def mean_gain(m, b, likelihood_ratio):
        p = prior * likelihood_ratio / (1 - prior + prior * likelihood_ratio)
        h, l = deviation_gains(b, badge_prize(m - 1, contract), contract)
        return p * h + (1 - p) * l

    bound = mean_gain(orders, badge, 1)
    for i in range(1, orders):
        possible = (mean_gain(orders - i, b, ratio**i)
                    for b in (-1, 1) for ratio in (low, high))
        bound += max(0, *possible)
    return bound


def verify_original_horizons():
    q0 = Fraction(16, 25)
    for n in range(1, 9):
        for m, b in [(n, -1), *product(range(1, n), (-1, 1))]:
            bound = first_trial_bound(m, b, q0, TABLE_3)
            if (m, b) == (1, -1):
                assert bound == 0
            else:
                assert bound < 0
            assert first_trial_bound(m, b, Fraction(1), TABLE_3) < 0
    # At length nine the analytic sufficient condition becomes inconclusive.
    assert first_trial_bound(8, 1, q0, TABLE_3) > 0
    upper = Fraction(9584041984, 14247436759)
    # Any fixed plan forced to try now has strictly negative H surplus and
    # earns d_L E_L[K] > 0 in L. Its expected surplus falls strictly with q.
    assert first_order_values(8, upper) == (0, 0)
    assert first_order_values(8, Fraction(33, 50))[1] > 0
    print(f"Section 9.5: posterior-range bounds certify horizons 1–8. At eight "
          f"orders strict elimination spans {100*float(upper-q0):.2f} percentage points.")

def verify_long_contract_test(contract):
    """Check the four gains against stationary profits and finite-horizon policy trees."""
    s_a = contract.probabilities(0, 1)[0]
    s_b = contract.probabilities(1, -1)[0]
    prize = badge_prize(1, contract) / (1 - s_a + s_b)
    usual_rate = contract.margin * s_b - contract.fast_cost + s_b * prize
    gains = {b: deviation_gains(b, prize, contract) for b in (-1, 1)}
    best_rates = []
    for model in (0, 1):
        rates = []
        for grade_a, grade_b in product((0, 1), repeat=2):
            a = contract.probabilities(grade_a, 1)[model]
            b = contract.probabilities(grade_b, -1)[model]
            share = b / (1 - a + b)
            rate = (share * (contract.margin * a - contract.fast_cost * grade_a)
                    + (1 - share) * (contract.margin * b - contract.fast_cost * grade_b))
            surplus = share * gains[1][model] if grade_a == 1 else 0
            if grade_b == 0:
                surplus += (1 - share) * gains[-1][model]
            assert rate - usual_rate == surplus
            rates.append(rate)
        best_rates.append(max(rates))
        assert (max(rates) > usual_rate) == any(gains[b][model] > 0 for b in (-1, 1))
    if max(best_rates) == usual_rate:
        # The best known-model value is an upper bound even for a Bayesian
        # vendor. Check all contingent plans at short horizons in both models.
        for m in (1, 2, 3):
            for b in (-1, 1):
                plans = policy_gains(m, b, True, contract)
                assert max(max(h, l) for _, h, l in plans) <= abs(prize)
    return prize, gains, usual_rate, best_rates


def policy_gains(orders, badge, transfer, contract):
    """Enumerate short decision trees. Keep a deviating plan even if its two gains are zero."""
    if orders == 0:
        return [(False, Fraction(0), Fraction(0))]
    wins = policy_gains(orders - 1, 1 if transfer else badge, transfer, contract)
    losses = policy_gains(orders - 1, -1 if transfer else badge, transfer, contract)
    prize = badge_prize(orders - 1, contract) if transfer else 0
    plans = []
    for grade in (0, 1):
        deviation = grade != int(badge == -1)
        gain_h, gain_l = deviation_gains(badge, prize, contract) if deviation else (0, 0)
        high, low = contract.probabilities(grade, badge)
        for (dev_w, h_w, l_w), (dev_l, h_l, l_l) in product(wins, losses):
            plans.append((deviation or dev_w or dev_l,
                          gain_h + high * h_w + (1 - high) * h_l,
                          gain_l + low * l_w + (1 - low) * l_l))
    return plans


def elimination_interval(orders, contract):
    """Apply Proposition 4 by exhaustive enumeration for short contracts (integer number of orders)."""
    if orders < 1:
        raise ValueError("The contract must contain at least one order.")
    d_h, d_l = deviation_gains(-1, 0, contract)
    if not d_h < 0 < d_l:
        raise ValueError("Section 9.4 assumes standard with B featured gains in L and loses in H.")
    fixed = policy_gains(orders, -1, False, contract)
    lower = Fraction(0)
    upper = max(low / -high for deviates, high, low in fixed if deviates)
    for deviates, high, low in policy_gains(orders, -1, True, contract):
        if not deviates:
            continue
        if high < 0:
            lower = max(lower, -low / high)
        elif high > 0:
            upper = min(upper, -low / high)
        elif low >= 0:
            return None
    if lower >= upper:
        return None
    return lower / (1 + lower), upper / (1 + upper)


def verify_general_contracts():
    # Values in the second example follow from a grade odds multiplier of
    # 121/79 in H and 119/81 in L at either badge position.
    def example_probabilities(grade, badge):
        if badge == -1:
            return (Fraction(1, 2),) * 2 if grade else (Fraction(79, 200), Fraction(81, 200))
        return (Fraction(484, 563), Fraction(476, 557)) if grade else (Fraction(4, 5),) * 2

    example = Contract(100, 10, example_probabilities)
    assert deviation_gains(-1, 0, example) == (Fraction(-1, 2), Fraction(1, 2))
    assert max(deviation_gains(1, 0, example)) < 0
    # For k >= 1, D_k increases from 40 towards 400/7. Gains are affine in D_k.
    assert badge_prize(1, example) == 40
    persistence = example.probabilities(0, 1)[0] - example.probabilities(1, -1)[0]
    assert persistence == Fraction(3, 10)
    limit = badge_prize(1, example) / (1 - persistence)
    assert limit == Fraction(400, 7)
    worst_gain = max(gain for prize in (badge_prize(1, example), limit) for badge in (-1, 1)
                     for gain in deviation_gains(badge, prize, example))
    assert worst_gain == -Fraction(350, 563) < -Fraction(1, 2)
    assert elimination_interval(2, example) == (Fraction(1, 2), Fraction(281, 560))
    # A separate exact six-order Bellman calculation checks expansion beyond five orders.
    six_cutoff = Fraction(404121211801, 799666559600)
    knots = prior_partition(6, -1, False, example)
    assert six_cutoff in knots and six_cutoff > Fraction(281, 560)
    for prior in knots:
        gap = first_grade_gap(6, prior, -1, False, example)
        if prior == six_cutoff:
            assert gap == 0
        else:
            assert gap * (six_cutoff - prior) > 0
    for m, b in [(6, -1)] + list(product(range(1, 6), (-1, 1))):
        lower = Fraction(1, 2)
        knots = prior_partition(m, b, True, example)
        for prior in [lower] + [point for point in knots if point > lower]:
            margin = b * first_grade_gap(m, prior, b, True, example)
            if (m, b, prior) == (1, -1, lower):
                assert margin == 0
            else:
                assert margin > 0
    # At 51%, buy two trial observations and continue with standard only after two wins.
    q = Fraction(51, 100)
    t_h, t_l = example.probabilities(0, -1)
    k_h, k_l = (2 + 98 * p**2 for p in (t_h, t_l))
    assert ((1 - q) * k_l - q * k_h) / 2 == Fraction(38351, 2000000) > 0
    # Proposition 5 separates the two published contracts using four numbers.
    prize, gains, rate, best = verify_long_contract_test(TABLE_3)
    assert prize == 35 and rate == 73 and gains[-1][1] == 2
    assert gains[-1][0] < 0 and max(gains[1]) < 0
    assert best == [73, Fraction(1250, 17)]
    low_standard = TABLE_3.probabilities(0, -1)[1]
    low_featured = TABLE_3.probabilities(0, 1)[1]
    share_b = (1 - low_featured) / (1 - low_featured + low_standard)
    assert share_b == Fraction(9, 34)
    assert Fraction(17, 50) * share_b * gains[-1][1] == Fraction(9, 50)
    prize, gains, rate, best = verify_long_contract_test(example)
    assert prize == Fraction(400, 7) and best == [rate, rate]
    assert all(max(pair) < 0 for pair in gains.values())

    # Equality is on the bounded side. A terminal trial can nevertheless pay.
    def boundary_probabilities(grade, badge):
        if (grade, badge) == (0, -1):
            return Fraction(1, 2), Fraction(73, 135)
        return TABLE_3.probabilities(grade, badge)

    boundary = Contract(100, 8, boundary_probabilities)
    prize, gains, rate, best = verify_long_contract_test(boundary)
    assert prize == 35 and gains[-1][1] == 0 and best == [rate, rate]
    assert first_grade_gap(1, Fraction(1, 10), -1, True, boundary) > 0
    print("Proposition 5: stationary profit differences match the badge-adjusted gains. "
          "Table 3's pilot earns €0.18 per order in the limit at 66%. The second contract "
          "and the zero-gain boundary have bounded surplus.")
    # The uniform sufficient condition is stronger than Proposition 4.
    assert deviation_gains(-1, badge_prize(1, TABLE_3), TABLE_3)[1] > 0

    def profitable_fast_when_featured(grade, badge):
        if grade == 1 and badge == 1:
            return Fraction(49, 50), Fraction(24, 25)
        return example_probabilities(grade, badge)

    empty_case = Contract(100, 10, profitable_fast_when_featured)
    assert elimination_interval(3, empty_case) is None
    for contract in (example, TABLE_3, empty_case):
        for n in range(1, 4):
            bounds = elimination_interval(n, contract)
            states = [(n, -1)] + list(product(range(1, n), (-1, 1)))
            points = set(prior_partition(n, -1, False, contract))
            for m, b in states:
                points.update(prior_partition(m, b, True, contract))
            if bounds:
                points.update(bounds)
            knots = sorted(points)
            probes = points | {(a + b) / 2 for a, b in zip(knots, knots[1:])}
            for q in probes:
                # Independent Bellman recursion updates the posterior after every
                # possible deviation. Test all states reached by the uninformative plan.
                silent = all(b * first_grade_gap(m, q, b, True, contract) > 0 for m, b in states)
                eliminated = 0 < q < 1 and silent and first_grade_gap(n, q, -1, False, contract) > 0
                assert eliminated == (bounds is not None and bounds[0] < q < bounds[1])
    print("Section 9.4: policy trees and Bayesian recursion agree on short contracts, including "
          "ties and an empty interval. The uniform badge-prize bound holds; with two orders the "
          "second example gives 1/2 < prior < 281/560. A 100-order fixed-badge trial plan "
          "earns strictly positive surplus at a 51% prior.")


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
    verify_general_contracts()
    verify_original_horizons()


if __name__ == "__main__":
    main()
