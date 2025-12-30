"""
Metrics for measuring path dependence effects.

This module provides:
- Mutual information estimation between work transcripts and rewards
- Concentration metrics (top-k share, Gini coefficient)

The key insight: in path-dependent systems, I(V;R) → 0 as T grows,
even when work has a local effect (lambda > 0) and transcripts vary.
"""

import numpy as np
from typing import Sequence

from shadow_futures.process import AgentState, SimulationResult


def estimate_mutual_information(
    transcripts: np.ndarray,
    rewards: np.ndarray,
) -> float:
    """
    Estimate mutual information I(V;R) using plug-in estimator.
    
    Uses the standard formula:
        I(V;R) = sum_{v,r} p(v,r) * log2(p(v,r) / (p(v) * p(r)))
    
    Both V and R are assumed discrete.
    
    Args:
        transcripts: Array of discrete transcript values (e.g., 0 or 1)
        rewards: Array of discrete reward outcomes (e.g., 0 or 1)
    
    Returns:
        Estimated mutual information in bits (log base 2)
    
    Note:
        This is a plug-in estimator with positive bias O(1/N) for small
        samples. When V and R are independent, small positive MI values
        reflect sampling variation and should shrink as N increases.
    """
    n = len(transcripts)
    if n == 0:
        return 0.0
    
    if len(rewards) != n:
        raise ValueError("transcripts and rewards must have same length")
    
    v_vals = np.unique(transcripts)
    r_vals = np.unique(rewards)
    
    mi = 0.0
    for v in v_vals:
        p_v = np.mean(transcripts == v)
        for r in r_vals:
            p_r = np.mean(rewards == r)
            p_vr = np.mean((transcripts == v) & (rewards == r))
            if p_vr > 0:
                mi += p_vr * np.log2(p_vr / (p_v * p_r))
    
    return float(max(0.0, mi))  # MI is non-negative; clamp numerical errors


def compute_concentration(
    agents: Sequence[AgentState],
    k: int = 1,
) -> float:
    """
    Compute top-k reward share concentration.
    
    Returns the fraction of total rewards held by the top k agents.
    This measures winner-take-all dynamics from path dependence.
    
    Args:
        agents: Sequence of AgentState objects
        k: Number of top agents to consider
    
    Returns:
        Fraction of total rewards held by top k agents
    """
    if not agents:
        return 0.0
    
    rewards = np.array([a.total_rewards for a in agents])
    total = rewards.sum()
    
    if total == 0:
        return 0.0
    
    # Sort descending and take top k
    sorted_rewards = np.sort(rewards)[::-1]
    top_k_sum = sorted_rewards[:k].sum()
    
    return top_k_sum / total


def compute_gini(agents: Sequence[AgentState]) -> float:
    """
    Compute Gini coefficient of reward distribution.
    
    Gini = 0 means perfect equality (all agents have same rewards)
    Gini = 1 means perfect inequality (one agent has all rewards)
    
    Under preferential attachment with alpha >= 1, Gini increases with T.
    
    Args:
        agents: Sequence of AgentState objects
    
    Returns:
        Gini coefficient in [0, 1]
    """
    if not agents:
        return 0.0
    
    rewards = np.array([a.total_rewards for a in agents], dtype=float)
    n = len(rewards)
    
    if n == 1:
        return 0.0
    
    total = rewards.sum()
    if total == 0:
        return 0.0
    
    # Sort ascending
    sorted_rewards = np.sort(rewards)
    
    # Gini formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    indices = np.arange(1, n + 1)
    gini = (2 * np.sum(indices * sorted_rewards)) / (n * total) - (n + 1) / n
    
    return max(0.0, min(1.0, gini))


def compute_ever_rewarded(agents: Sequence[AgentState]) -> np.ndarray:
    """
    Compute binary indicator of whether each agent ever received reward.
    
    Args:
        agents: Sequence of AgentState objects
    
    Returns:
        Binary array where 1 = received at least one reward
    """
    return np.array([1 if a.total_rewards > 0 else 0 for a in agents])


def compute_reward_thresholded(
    agents: Sequence[AgentState],
    threshold: int = 2,
) -> np.ndarray:
    """
    Compute binary indicator of whether each agent received >= threshold rewards.
    
    Using threshold >= 2 (instead of > 0) provides a more stable outcome
    variable for MI estimation, because it reduces "everyone gets at least
    one hit early" artifacts in reinforcement processes.
    
    Args:
        agents: Sequence of AgentState objects
        threshold: Minimum reward count to be considered "rewarded"
    
    Returns:
        Binary array where 1 = received at least threshold rewards
    """
    return np.array([1 if a.total_rewards >= threshold else 0 for a in agents])


def compute_mi_from_result(
    result: SimulationResult,
    reward_threshold: int = 2,
) -> float:
    """
    Compute mutual information I(V;R) between transcripts and reward status.
    
    R is defined as "received >= reward_threshold rewards by time T" (binary).
    V is the discrete work transcript (typically 0=low or 1=high).
    
    Using threshold >= 2 (default) provides a more stable outcome variable
    than "ever rewarded" (threshold=1), because it reduces artifacts from
    early-entry agents who get lucky once but never again.
    
    Key qualitative behavior: I(V;R) tends to shrink as T grows when
    alpha >= 1 and lambda < 1. Finite-sample fluctuations may cause
    non-monotonicity in small experiments.
    
    Note:
        When lambda=0, V is independent of allocation state, so I(V;R)
        should theoretically be zero. Small positive values reflect
        finite-sample bias in the plug-in estimator.
    
    Args:
        result: SimulationResult from a simulation run
        reward_threshold: Minimum rewards to count as "rewarded" (default: 2)
    
    Returns:
        Mutual information in bits (log base 2)
    """
    transcripts = np.array([a.transcript for a in result.agents])
    rewards = compute_reward_thresholded(result.agents, threshold=reward_threshold)
    return estimate_mutual_information(transcripts, rewards)


def compute_metrics_summary(result: SimulationResult) -> dict:
    """
    Compute a full summary of metrics from a simulation result.
    
    Args:
        result: SimulationResult from a simulation run
    
    Returns:
        Dictionary with all computed metrics
    """
    agents = result.agents
    
    return {
        "T": result.T,
        "alpha": result.alpha,
        "lambda_effect": result.lambda_effect,
        "seed": result.seed,
        "n_agents": len(agents),
        "total_rewards": sum(a.total_rewards for a in agents),
        "mi_v_r": compute_mi_from_result(result),
        "top_1_share": compute_concentration(agents, k=1),
        "top_10_share": compute_concentration(agents, k=10),
        "gini": compute_gini(agents),
        "fraction_ever_rewarded": np.mean(compute_ever_rewarded(agents)),
    }

