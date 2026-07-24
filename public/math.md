# The Central Shadow Futures Equations

Canonical page: https://shadow-futures.vercel.app/math

## Reinforced score

s_i,t(beta) = exp(beta x_i) (a + N_i(t))^rho

## Allocation probability

p_i,t(beta) = s_i,t(beta) / sum_j s_j,t(beta)

## Remaining competition

epsilon_t(beta) = 1 - max_i p_i,t(beta)

## Comparison budget

B_T(beta) = sum from t = 0 to T - 1 of epsilon_t(beta)

## Central implication

Under the paper's common-design, local-equivalence, two-sided Hellinger-control, and finite-comparison conditions:

B_infinity(beta) < infinity for every beta being compared

implies mutual absolute continuity of the corresponding complete-history laws, which implies that no estimator based on one realized market history can universally and consistently recover every nonconstant contribution functional F(beta).

## Symbols

- x_i: agent i's verified contribution-related input.
- beta: how strongly that input affects the chance of winning the next opportunity.
- N_i(t): attention, customers, sales, or another advantage agent i has accumulated.
- rho: reinforcement strength.
- p_i,t: probability that agent i receives the next opportunity.
- epsilon_t: probability left for an agent other than the current favorite.
- B_T: cumulative comparison through time T.

## Interpretation

The result does not say contribution has no causal effect. It says the one reinforced history may not contain enough comparison to recover the magnitude of that effect.

Full paper: https://shadow-futures.vercel.app/paper
