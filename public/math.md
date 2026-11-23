# The Shadow Futures Mathematics

Canonical page: https://shadow-futures.vercel.app/math

## From winning odds to the learning limit

The paper's general allocation model uses verified input profiles x_i,t and accumulated position s_i,t. The parameter beta describes the direct input effect. The positive weight w is shorthand:

    w_i,t(beta) = exp(x_i,t^T beta + s_i,t)
    p_i,t(beta) = w_i,t(beta) / sum_j w_j,t(beta)

Probabilities are conditional on the observed history and add up to one. A better input can improve winning odds even when accumulated position also matters.

## Remaining competition

    epsilon_t(beta) = 1 - max_i p_i,t(beta)

This is the chance that the next reward goes to someone other than the current favorite. The comparison budget adds these chances:

    B_T(beta) = sum from t = 0 to T - 1 of epsilon_t(beta)
    B_infinity(beta) = sum from t = 0 to infinity of epsilon_t(beta)

For 1,000 rounds at hypothetical constant odds, a two-person contest at 50–50 adds 500 to the budget. A favorite with a 99.9% chance adds only 1. These are sums of probabilities, not counts of observed upsets or independent experiments.

## Central implication

The finite-budget condition is:

    P_beta(B_infinity(beta) < infinity) = 1
    for every beta in Theta

It must hold with probability one under every parameter, together with:

- A shared initial law and observed-history space; input profiles and position are common predictable functions of the past.
- One-step laws with the same possible recipients after each history, for every parameter pair (“local equivalence”).
- For each parameter pair, at every date and history, Hellinger separation bounded by a finite constant times residual contestability in both parameter directions.
- Inclusion of all parameter-dependent information in other observed variables.

Theorem 1 then gives mutually absolutely continuous complete-history laws P_beta and P_beta' for every beta, beta' in Theta. For any chosen nonconstant contribution functional F(beta), no estimator based on one history can be consistent at every beta.

“Consistent” means its errors tend to zero as the observed history grows. Observations can favor one parameter over another; the theorem rules out a method that works consistently at every parameter.

Shrinking chances alone are insufficient: 1/2 + 1/3 + 1/4 + … is infinite. Finite total comparison is stronger: 1/2 + 1/4 + 1/8 + … equals 1.

## The simulation is a special case

The app uses fixed inputs and polynomial feedback:

    s_i,t = rho log(a + N_i(t))
    w_i,t(beta) = exp(beta x_i) (a + N_i(t))^rho

N_i(t) counts previous rewards; a > 0 gives everyone a positive starting weight. With fixed inputs and a finite set of competitors, rho > 1 gives the paper's strong-reinforcement case: eventual allocation monopoly and a finite comparison budget with probability one. A finite simulation does not prove the infinite-history condition.

## Symbols

- x_i,t: person or firm i's verified input profile at time t, including work, quality, effort or capital at risk.
- beta: the parameter or set of weights describing the direct input effect on reward odds.
- s_i,t: accumulated position, such as reputation, ranking or prior sales.
- w_i,t: a positive allocation weight combining input effects and position.
- p_i,t: the conditional chance that i receives the next reward.
- epsilon_t: the chance left for someone other than the current favorite.
- B_T, B_infinity: comparison through time T, or over the entire history.
- F(beta): a chosen contribution quantity that changes with beta.
- P_beta, Theta: the law of histories under beta, and the parameter set under consideration.

## Interpretation

Mutual absolute continuity means the complete-history laws have the same probability-zero events. It does not mean the laws are identical.

Exactly identical observable laws arise in a separate model where unobserved initial position can offset a change in the input effect. Theorem 2 requires such equality of laws plus different contribution assignments on records with positive probability to rule out one exact contribution-versus-rent tax that works in both economies.

Independent markets with a shared beta can add information when their fixed-horizon observable laws distinguish parameters, under Appendix E.1's conditions. More routes or repeated transactions alone do not guarantee identification.

Neither theorem says productive inputs have no causal effect.

## Sources

Model: equation (1). Comparison budget: equations (2)–(3). Learning limit: Theorem 1 and Appendix B. Strong reinforcement: Appendix D. Replication, position invariance and taxation: Appendix E and sections 3–5.

- [Full paper](https://shadow-futures.vercel.app/paper)
- [Paper PDF](https://shadow-futures.vercel.app/paper.pdf)
- [Methodology and assumptions](https://shadow-futures.vercel.app/methodology)
