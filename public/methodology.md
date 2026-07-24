# Shadow Futures Methodology

Canonical page: https://shadow-futures.vercel.app/methodology

## Question

When a self-reinforcing market closes off alternative paths, does one realized history still contain enough comparison to recover contribution?

## Allocation model

Each competitor receives a score combining a verified contribution-related input and accumulated advantage:

s_i,t(beta) = exp(beta x_i) (a + N_i(t))^rho

The market converts those scores into probabilities:

p_i,t(beta) = s_i,t(beta) / sum_j s_j,t(beta)

## Comparison budget

Contest openness at time t is:

epsilon_t(beta) = 1 - max_i p_i,t(beta)

Cumulative comparison through time T is:

B_T(beta) = sum from t = 0 to T - 1 of epsilon_t(beta)

A transaction is informative only when more than one competitor has a meaningful chance. A high transaction count can therefore coexist with little new comparison.

## Theorem conditions

- The design is common and predictable from the same observed past.
- Nearby parameter values have locally equivalent one-step laws.
- Hellinger separation in both directions is controlled by remaining comparison.
- Total comparison is finite under every parameter being compared.
- Any additional observed process carrying parameter-dependent information is included.

The conclusion is mutual absolute continuity of complete-history laws, not equality of distributions.

## Result

Under those conditions, finite total comparison prevents universal consistent recovery of every nonconstant contribution functional from one realized market history. It also prevents tests from separating two contribution parameters with vanishing total error.

## Simulation boundary

The website's interactive models illustrate reinforced allocation and disappearing comparison. They are reproducible demonstrations, not empirical estimates or forecasts for any particular platform or industry.

## Full sources

- [Paper landing page](https://shadow-futures.vercel.app/paper)
- [Paper PDF](https://shadow-futures.vercel.app/paper.pdf)
- [Mathematics](https://shadow-futures.vercel.app/math)
- [Comparison Playground](https://shadow-futures.vercel.app/playground)
