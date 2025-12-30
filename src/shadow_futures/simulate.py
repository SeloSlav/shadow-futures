"""
Simulation runners for shadow futures experiments.

This module provides high-level functions to:
1. Demonstrate shadow futures: same work, different outcomes
2. Estimate mutual information collapse over T
3. Run parameter sweeps for analysis
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from shadow_futures.process import (
    PreferentialAttachmentProcess,
    SimulationResult,
    simulate_single_run,
)
from shadow_futures.metrics import (
    compute_metrics_summary,
    compute_mi_from_result,
    compute_concentration,
    compute_gini,
    compute_ever_rewarded,
)


@dataclass
class ShadowFuturesResult:
    """
    Result of a shadow futures experiment for a focal agent.
    
    Shows that identical work (same transcript) can yield divergent outcomes
    across different random seeds (realizations of path dependence).
    
    Attributes:
        focal_agent_index: Index of the focal agent
        n_simulations: Number of simulations run
        n_rewarded: How many simulations where focal agent got any reward
        n_unrewarded: How many simulations where focal agent got no reward
        reward_counts: Array of reward counts across simulations
        transcript: The focal agent's (identical) transcript value
        parameters: Dict of simulation parameters
    """
    focal_agent_index: int
    n_simulations: int
    n_rewarded: int
    n_unrewarded: int
    reward_counts: np.ndarray
    transcript: int
    parameters: dict


def run_shadow_futures_experiment(
    T: int = 100,
    alpha: float = 1.0,
    A0: float = 1.0,
    focal_agent_index: int = 0,
    n_simulations: int = 100,
    base_seed: int = 42,
    lambda_effect: float = 0.0,
) -> ShadowFuturesResult:
    """
    Demonstrate shadow futures for a focal agent.
    
    Runs multiple simulations with identical parameters but different seeds.
    For each simulation, tracks whether the focal agent received any rewards.
    
    The key insight: even with identical work transcripts (verified effort),
    the focal agent sometimes succeeds and sometimes fails, purely due to
    different realizations of the path-dependent process.
    
    Note:
        All agents are assigned transcript=0 (p_high=0.0) to ensure the
        statement "identical transcript across simulations" is literally true.
        Transcript heterogeneity is only used in the MI experiment.
    
    Args:
        T: Number of time steps
        alpha: Path dependence exponent
        A0: Initial attachment
        focal_agent_index: Which agent to track (0 = first entrant)
        n_simulations: Number of simulations to run
        base_seed: Base random seed (each sim uses base_seed + i)
        lambda_effect: Local work effect weight (typically 0 for this demo)
    
    Returns:
        ShadowFuturesResult with divergent outcome statistics
    """
    reward_counts = []
    
    for i in range(n_simulations):
        # Force p_high=0.0 so all agents have transcript=0 (identical work)
        result = simulate_single_run(
            T=T,
            alpha=alpha,
            A0=A0,
            lambda_effect=lambda_effect,
            p_high=0.0,  # All transcripts = 0
            seed=base_seed + i,
        )
        
        if focal_agent_index < len(result.agents):
            reward_counts.append(result.agents[focal_agent_index].total_rewards)
        else:
            reward_counts.append(0)
    
    reward_counts = np.array(reward_counts)
    n_rewarded = np.sum(reward_counts > 0)
    n_unrewarded = n_simulations - n_rewarded
    
    # Transcript is explicitly 0 for all agents in all runs
    transcript = 0
    
    return ShadowFuturesResult(
        focal_agent_index=focal_agent_index,
        n_simulations=n_simulations,
        n_rewarded=int(n_rewarded),
        n_unrewarded=int(n_unrewarded),
        reward_counts=reward_counts,
        transcript=transcript,
        parameters={
            "T": T,
            "alpha": alpha,
            "A0": A0,
            "lambda_effect": lambda_effect,
            "base_seed": base_seed,
        },
    )


@dataclass
class MIExperimentResult:
    """
    Result of mutual information experiment across T values.
    
    Attributes:
        T_values: Array of T values tested
        alphas: Array of alpha values tested
        lambdas: Array of lambda values tested
        mi_matrix: 3D array of mean MI values [T_idx, alpha_idx, lambda_idx]
        mi_std_matrix: 3D array of MI std dev [T_idx, alpha_idx, lambda_idx]
        concentration_matrix: 3D array of mean top-1 concentration
        concentration_std_matrix: 3D array of top-1 concentration std dev
        gini_matrix: 3D array of mean Gini coefficients
        gini_std_matrix: 3D array of Gini std dev
        n_runs_per_point: Number of runs averaged per data point
    
    Note:
        R is defined as "received >= 2 rewards by time T" (binary indicator).
        Using threshold >= 2 reduces early-entry artifacts.
        MI is measured in bits (log base 2).
        When lambda=0, MI should be approximately zero; small positive
        values reflect finite-sample sampling variation.
    """
    T_values: np.ndarray
    alphas: np.ndarray
    lambdas: np.ndarray
    mi_matrix: np.ndarray
    mi_std_matrix: np.ndarray
    concentration_matrix: np.ndarray
    concentration_std_matrix: np.ndarray
    gini_matrix: np.ndarray
    gini_std_matrix: np.ndarray
    n_runs_per_point: int


def run_mi_experiment(
    T_values: Optional[list[int]] = None,
    alphas: Optional[list[float]] = None,
    lambdas: Optional[list[float]] = None,
    n_runs: int = 50,
    base_seed: int = 42,
    A0: float = 1.0,
    p_high: float = 0.5,
    v_low: float = 0.1,
    v_high: float = 0.9,
) -> MIExperimentResult:
    """
    Run experiment to demonstrate mutual information collapse.
    
    Sweeps over T (system size), alpha (path dependence strength),
    and lambda (local work effect). For each combination, runs multiple
    simulations and computes average MI between transcripts and rewards.
    
    Key qualitative behavior: For alpha >= 1 and lambda < 1, MI tends to
    shrink as T increases, while concentration increases. Finite-sample
    fluctuations may cause non-monotonicity in small experiments.
    
    Args:
        T_values: List of T values to test
        alphas: List of alpha values to test
        lambdas: List of lambda values to test
        n_runs: Number of runs per parameter combination
        base_seed: Base random seed
        A0: Initial attachment
        p_high: Probability of high transcript
        v_low: Low transcript value
        v_high: High transcript value
    
    Returns:
        MIExperimentResult with MI, concentration, and Gini matrices
    """
    if T_values is None:
        T_values = [50, 100, 200, 500]
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.3, 0.5]
    
    T_arr = np.array(T_values)
    alpha_arr = np.array(alphas)
    lambda_arr = np.array(lambdas)
    
    n_T = len(T_values)
    n_alpha = len(alphas)
    n_lambda = len(lambdas)
    
    mi_matrix = np.zeros((n_T, n_alpha, n_lambda))
    mi_std_matrix = np.zeros((n_T, n_alpha, n_lambda))
    concentration_matrix = np.zeros((n_T, n_alpha, n_lambda))
    concentration_std_matrix = np.zeros((n_T, n_alpha, n_lambda))
    gini_matrix = np.zeros((n_T, n_alpha, n_lambda))
    gini_std_matrix = np.zeros((n_T, n_alpha, n_lambda))
    
    for i_T, T in enumerate(T_values):
        for i_a, alpha in enumerate(alphas):
            for i_l, lam in enumerate(lambdas):
                mi_vals = []
                conc_vals = []
                gini_vals = []
                
                for run in range(n_runs):
                    # Use same run seeds across parameter cells for fair comparison
                    # Each run uses base_seed + run, independent of parameter indices
                    seed = base_seed + run
                    result = simulate_single_run(
                        T=T,
                        alpha=alpha,
                        A0=A0,
                        lambda_effect=lam,
                        p_high=p_high,
                        v_low=v_low,
                        v_high=v_high,
                        seed=seed,
                    )
                    
                    mi_vals.append(compute_mi_from_result(result))
                    conc_vals.append(compute_concentration(result.agents, k=1))
                    gini_vals.append(compute_gini(result.agents))
                
                mi_matrix[i_T, i_a, i_l] = np.mean(mi_vals)
                mi_std_matrix[i_T, i_a, i_l] = np.std(mi_vals)
                concentration_matrix[i_T, i_a, i_l] = np.mean(conc_vals)
                concentration_std_matrix[i_T, i_a, i_l] = np.std(conc_vals)
                gini_matrix[i_T, i_a, i_l] = np.mean(gini_vals)
                gini_std_matrix[i_T, i_a, i_l] = np.std(gini_vals)
    
    return MIExperimentResult(
        T_values=T_arr,
        alphas=alpha_arr,
        lambdas=lambda_arr,
        mi_matrix=mi_matrix,
        mi_std_matrix=mi_std_matrix,
        concentration_matrix=concentration_matrix,
        concentration_std_matrix=concentration_std_matrix,
        gini_matrix=gini_matrix,
        gini_std_matrix=gini_std_matrix,
        n_runs_per_point=n_runs,
    )


def run_single_simulation_summary(
    T: int = 100,
    alpha: float = 1.0,
    A0: float = 1.0,
    lambda_effect: float = 0.0,
    p_high: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Run a single simulation and return a summary dict.
    
    Useful for CLI output and JSON export.
    
    Args:
        T: Number of time steps
        alpha: Path dependence exponent
        A0: Initial attachment
        lambda_effect: Local work effect weight
        p_high: Probability of high transcript
        seed: Random seed
    
    Returns:
        Dictionary with simulation summary
    """
    result = simulate_single_run(
        T=T,
        alpha=alpha,
        A0=A0,
        lambda_effect=lambda_effect,
        p_high=p_high,
        seed=seed,
    )
    return compute_metrics_summary(result)

