"""
Shadow Futures: Demonstrating why verifiable work cannot signal value
in path-dependent economies with preferential attachment.

This package provides simulation tools to empirically demonstrate that:
1. Identical verified work can yield divergent reward outcomes
2. Mutual information between work and reward collapses as path dependence grows
3. Shadow futures exist: unrealized trajectories with identical work but different outcomes
"""

__version__ = "0.1.0"

from shadow_futures.process import (
    PreferentialAttachmentProcess,
    simulate_single_run,
    AgentState,
)
from shadow_futures.simulate import (
    run_shadow_futures_experiment,
    run_mi_experiment,
)
from shadow_futures.metrics import (
    estimate_mutual_information,
    compute_concentration,
    compute_gini,
    compute_reward_thresholded,
)

__all__ = [
    "PreferentialAttachmentProcess",
    "simulate_single_run",
    "AgentState",
    "run_shadow_futures_experiment",
    "run_mi_experiment",
    "estimate_mutual_information",
    "compute_concentration",
    "compute_gini",
    "compute_reward_thresholded",
]

