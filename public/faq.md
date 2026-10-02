# Shadow Futures FAQ

Canonical page: https://shadow-futures.vercel.app/faq

## What are shadow futures?

Shadow futures are alternative market histories that had a positive chance of happening under the same inputs, rules, and starting conditions, but with different allocation shocks and final rewards. Imagine two equally equipped bakeries: an early order gives one reviews and visibility that help it win later orders. The history in which the other got that first order is a shadow future. This illustrates feedback, not a finding about actual bakeries.

## Does the theory say work, talent, quality, or risk do not matter?

No. These inputs can be real, observed, and directly affect reward probabilities. The question is whether one market history contains enough evidence to estimate the causal difference they made, including later feedback. A large prize may also encourage useful experimentation without measuring the winner's contribution exactly.

## Why is a large transaction count not enough?

When one alternative is almost certain to win, the model limits how much information the next allocation adds about productive inputs. At a constant 99.9% chance for the leader, 1,000 rounds add 1 to the comparison budget: 1,000 × 0.001. That does not mean exactly one rival wins. Nor does this example prove a finite lifetime budget: a fixed small chance added forever sums to infinity.

## What is the comparison budget?

At each round, take the probability that someone other than the current favorite receives the next reward. Add those probabilities across rounds. The favorite may change. This is a sum of model probabilities, not a count of rival wins. It bounds information in the paper's model; a large or infinite budget does not alone guarantee learning.

## What exactly does the main theorem prove?

Under its conditions, no one method can learn a contribution measure reliably across all candidate explanations from one growing history when that measure differs between them. No test between two input effects can make both kinds of mistake vanish. Some evidence and partial learning remain possible.

The conditions include common input and position rules applied to the recorded past, the same possible next recipients, statistical differences between predictions bounded by a fixed multiple of the chance of choosing outside the leader, and a finite lifetime comparison budget under every candidate effect. Extra observations whose probabilities depend on the effect must have their information counted separately. Different input effects need not predict identical distributions.

## Is this just preferential attachment or increasing returns?

No. Those describe mechanisms by which leads compound. Shadow Futures studies what a loss of comparison means for estimating contribution. In the paper's reinforced model, sufficiently strong feedback produces eventual allocation monopoly and a finite comparison budget. Linear preferential attachment and concentration alone do not establish the theorem's conditions.

## Does inequality prove the theorem?

No. A Lorenz curve or final market-share distribution is like a scoreboard: it describes how rewards ended up divided. It does not by itself identify what caused them or establish the theorem's conditions.

## Is hidden starting advantage the same problem?

No. The paper also gives a latent-position model in which changing the input effect and offsetting hidden starting advantage leaves every observable probability identical. That is a separate identification problem. More copies of the same unidentified design do not automatically resolve it.

## What can preserve learning?

Randomized exposure, independent trials, portability, multihoming, interoperability, structural separation, independent procurement, and public options can preserve alternative paths. They must create informative variation. More platform names or a reset alone do not guarantee independence or learning. The appendix's replication result also requires that different contribution parameters predict different observable outcomes.

## Does this make every merger harmful?

No. The paper asks whether a merger removes independent routes through which rivals receive exposure and build records. Preserving those routes can have informational value, but welfare also depends on decision errors and the cost of preserving independent channels.

## What does the result imply for taxation and redistribution?

The separate tax theorem says that, when two economies produce identical observable laws but assign different contributions to a reward, one rule using that record cannot tax exactly the residual rent in both. It does not calculate an individual's luck percentage or prove all high income is rent.

The paper discusses taxing observable compounding and an unconditional social dividend that does not reconstruct an exact merit ranking. Progressive taxation or UBI also require policy goals, funding choices, and evidence about incentives and costs. The theorem determines no tax rate or uniquely correct transfer policy.

## Does this apply to AI agents and automated payments?

Potentially, as an application beyond the paper. Automated payments may extend a reinforced purchasing path rather than provide independent tests of competing firms. The relevant questions are whether informative alternative allocations remain possible and whether the theorem's conditions hold. Payment volume or a protocol alone establishes neither.

## Where is the proof?

- [Paper landing page](https://shadow-futures.vercel.app/paper)
- [Searchable PDF](https://shadow-futures.vercel.app/paper.pdf)
- [Methodology](https://shadow-futures.vercel.app/methodology)
- [Mathematics](https://shadow-futures.vercel.app/math)
