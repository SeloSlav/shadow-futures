# Shadow Futures Methodology

Canonical page: https://shadow-futures.vercel.app/methodology

## Question

Work, quality and risk can improve winning odds, while an early win can improve later access. Can one resulting market history reveal how much the productive inputs contributed?

Here, contribution is the difference an input would make to rewards under a specified counterfactual, including later feedback. Verifying that work happened is different from estimating that effect.

## Allocation model

The paper's general conditional-logit model combines verified input profiles x_i,t with accumulated-position indices s_i,t. The parameter beta describes the direct input effect. Using w as shorthand for a positive allocation weight:

    w_i,t(beta) = exp(x_i,t^T beta + s_i,t)
    p_i,t(beta) = w_i,t(beta) / sum_j w_j,t(beta)

The app illustrates a special case with fixed inputs and polynomial feedback:

    s_i,t = rho log(a + N_i(t))
    p_i,t(beta) = exp(beta x_i) (a + N_i(t))^rho
                  / sum_j exp(beta x_j) (a + N_j(t))^rho

N_i(t) counts earlier rewards; a is a positive starting value; rho controls reinforcement. This special case is not the general theorem's only allocation rule.

## Comparison budget

Residual contestability is the chance that the next reward goes to someone other than the current favorite:

    epsilon_t(beta) = 1 - max_i p_i,t(beta)
    B_T(beta) = sum from t = 0 to T - 1 of epsilon_t(beta)
    B_infinity(beta) = sum from t = 0 to infinity of epsilon_t(beta)

For 1,000 hypothetical constant-odds allocations:

- A two-person contest at 50–50 adds 1,000 × 0.5 = 500 to the budget.
- A favorite at 99.9% adds 1,000 × 0.001 = 1.

The budget adds probabilities. It is not the observed number of upsets or a count of independent experiments.

With input-profile distances bounded by D_X, Proposition 1 gives tr I_t(beta) <= D_X^2 epsilon_t(beta). This is an upper bound on conditional information about beta, not an equality. More comparison permits more information but does not guarantee it: the inputs must also differ informatively.

## Theorem conditions

- All parameters share the same initial law and observed-history space. Input profiles and state indices are common predictable functions of the observed past.
- For every parameter pair, one-step laws have the same possible recipients after each history. This is “local equivalence”; it does not mean only nearby parameter values.
- For each parameter pair, at every date and history, one-step Hellinger separation is bounded by a finite constant times residual contestability, in both parameter directions.
- B_infinity(beta) is finite with probability one under P_beta, for every beta in the parameter set Theta.
- All parameter-dependent information in other observed variables must be included in the experiment.

The conclusion is mutual absolute continuity of complete-history laws, not equality of distributions.

## Result

Under those conditions, complete-history laws P_beta and P_beta' are mutually absolutely continuous for every beta, beta' in Theta. They have the same probability-zero events, but may assign different probabilities to possible outcomes.

For any chosen nonconstant contribution functional F(beta), no estimator based on one history can be consistent at every beta: its errors cannot tend to zero at every parameter as the history grows. Observations can still favor one parameter over another.

The theorem also rules out tests whose two error probabilities both vanish, and confidence sets that shrink to a point while coverage tends to one at every parameter.

## Shrinking is different from finite

Chances of 1/2, 1/3, 1/4, … approach zero, but sum to infinity. Chances of 1/2, 1/4, 1/8, … sum to 1. The theorem requires a finite total with probability one under every parameter, not merely shrinking chances.

In the paper's finite-agent, fixed-input model with polynomial feedback g(z) = z^rho, rho > 1 implies eventual allocation monopoly and a finite budget with probability one. A concentrated chart at a finite date does not prove that limiting result.

## Simulation boundary

The story fixes 24 creators' input multipliers between 0.84 and 1.18 and allocates 1,600 recommendations with feedback strength 1.55. Fixed seeds make replays reproducible. These are illustrations, not empirical estimates, forecasts, or proofs of an infinite-history condition.

## Keeping learning alive

Repeated randomized exposure can preserve a chance outside the favorite. Independent starts and distinct routes to audiences can add evidence; several storefronts sharing one ranking need not be independent paths.

Appendix E.1's replication result requires independent markets with a common contribution parameter whose fixed-horizon observable laws distinguish parameters, under the paper's compact-parameter-set conditions. More channels alone do not guarantee identification. Random exposure changes the allocation rule.

## Policy boundaries

Competition can preserve comparison histories, but the paper does not conclude that every merger is harmful. Preserving independent routes has costs as well as potential information benefits.

Theorem 2 addresses a separate condition: two structural economies have exactly the same observable law but assign different contribution to the same reward on records with positive probability. A tax based on the observed record must be the same in both; it cannot equal reward minus contribution in both.

Exact equality of observable laws can arise when unobserved initial position offsets a change in the input effect. This is distinct from Theorem 1's equivalent, potentially different laws. Neither result says work is irrelevant or all income is rent, and neither selects a tax rate.

## Full sources

Model and information: sections 2–3, Proposition 1, Appendices A–C. Single-history theorem: Theorem 1 and Appendix B. Strong reinforcement: Appendix D. Replication, position invariance and taxation: sections 3–5 and Appendix E.

- [Paper landing page](https://shadow-futures.vercel.app/paper)
- [Paper PDF](https://shadow-futures.vercel.app/paper.pdf)
- [Mathematics](https://shadow-futures.vercel.app/math)
- [Comparison Playground](https://shadow-futures.vercel.app/playground)
