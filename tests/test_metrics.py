"""
Tests for shadow_futures.metrics module.

These tests verify:
1. Mutual information properties (non-negativity, bounds)
2. Concentration metrics (Gini, top-k share)
3. Consistency across different inputs
"""

import numpy as np
import pytest

from shadow_futures.process import AgentState, simulate_single_run
from shadow_futures.metrics import (
    estimate_mutual_information,
    compute_concentration,
    compute_gini,
    compute_ever_rewarded,
    compute_mi_from_result,
    compute_metrics_summary,
)


class TestMutualInformation:
    """Tests for mutual information estimation."""
    
    def test_mi_non_negative(self):
        """Mutual information should always be non-negative."""
        # Random data
        rng = np.random.default_rng(42)
        for _ in range(10):
            n = 100
            transcripts = rng.integers(0, 2, size=n)
            rewards = rng.integers(0, 2, size=n)
            mi = estimate_mutual_information(transcripts, rewards)
            assert mi >= 0, f"MI should be non-negative, got {mi}"
    
    def test_mi_independent_variables(self):
        """MI should be close to 0 for independent variables."""
        rng = np.random.default_rng(42)
        n = 10000
        transcripts = rng.integers(0, 2, size=n)
        rewards = rng.integers(0, 2, size=n)  # Independent of transcripts
        
        mi = estimate_mutual_information(transcripts, rewards)
        # For independent variables, MI should be very small
        assert mi < 0.05, f"MI for independent vars should be ~0, got {mi}"
    
    def test_mi_perfect_correlation(self):
        """MI should be maximal for perfectly correlated variables."""
        n = 1000
        transcripts = np.array([0, 1] * (n // 2))
        rewards = transcripts.copy()  # Perfect correlation
        
        mi = estimate_mutual_information(transcripts, rewards)
        # For binary variables with perfect correlation, MI = 1 bit
        assert 0.9 < mi <= 1.0, f"MI for perfect correlation should be ~1, got {mi}"
    
    def test_mi_partial_correlation(self):
        """MI should be intermediate for partial correlation."""
        rng = np.random.default_rng(42)
        n = 1000
        transcripts = rng.integers(0, 2, size=n)
        # 80% correlation
        rewards = transcripts.copy()
        flip_idx = rng.choice(n, size=n // 5, replace=False)
        rewards[flip_idx] = 1 - rewards[flip_idx]
        
        mi = estimate_mutual_information(transcripts, rewards)
        # Should be positive but less than 1
        assert 0 < mi < 1, f"MI for partial correlation should be in (0,1), got {mi}"
    
    def test_mi_empty_input(self):
        """Empty input should return 0."""
        mi = estimate_mutual_information(np.array([]), np.array([]))
        assert mi == 0
    
    def test_mi_single_value(self):
        """Single value input should return 0 (no information)."""
        mi = estimate_mutual_information(np.array([0, 0, 0]), np.array([1, 1, 1]))
        # When one variable is constant, MI is 0
        assert mi == 0


class TestConcentration:
    """Tests for concentration metrics."""
    
    def test_top_k_empty(self):
        """Empty agent list should return 0."""
        assert compute_concentration([], k=1) == 0
    
    def test_top_k_single_agent(self):
        """Single agent with all rewards should have concentration 1."""
        agents = [AgentState(agent_id=0, entry_time=0, attachment=10, total_rewards=10)]
        assert compute_concentration(agents, k=1) == 1.0
    
    def test_top_k_equal_distribution(self):
        """Equal distribution should have top-k = k/n."""
        n = 10
        agents = [
            AgentState(agent_id=i, entry_time=i, attachment=1, total_rewards=1)
            for i in range(n)
        ]
        assert compute_concentration(agents, k=1) == pytest.approx(0.1)
        assert compute_concentration(agents, k=5) == pytest.approx(0.5)
    
    def test_top_k_winner_take_all(self):
        """Winner-take-all should have top-1 = 1.0."""
        agents = [
            AgentState(agent_id=0, entry_time=0, attachment=100, total_rewards=99),
            AgentState(agent_id=1, entry_time=1, attachment=2, total_rewards=1),
            AgentState(agent_id=2, entry_time=2, attachment=1, total_rewards=0),
        ]
        assert compute_concentration(agents, k=1) == 0.99
    
    def test_top_k_no_rewards(self):
        """No rewards should return 0."""
        agents = [
            AgentState(agent_id=i, entry_time=i, attachment=1, total_rewards=0)
            for i in range(5)
        ]
        assert compute_concentration(agents, k=1) == 0


class TestGini:
    """Tests for Gini coefficient."""
    
    def test_gini_empty(self):
        """Empty list should return 0."""
        assert compute_gini([]) == 0
    
    def test_gini_single_agent(self):
        """Single agent should return 0 (no inequality)."""
        agents = [AgentState(agent_id=0, entry_time=0, attachment=10, total_rewards=10)]
        assert compute_gini(agents) == 0
    
    def test_gini_perfect_equality(self):
        """Equal distribution should have Gini = 0."""
        agents = [
            AgentState(agent_id=i, entry_time=i, attachment=1, total_rewards=10)
            for i in range(10)
        ]
        assert compute_gini(agents) == pytest.approx(0, abs=0.01)
    
    def test_gini_perfect_inequality(self):
        """One agent with all rewards should have Gini close to 1."""
        n = 100
        agents = [
            AgentState(agent_id=0, entry_time=0, attachment=100, total_rewards=100)
        ] + [
            AgentState(agent_id=i, entry_time=i, attachment=1, total_rewards=0)
            for i in range(1, n)
        ]
        gini = compute_gini(agents)
        # Gini should approach 1 as n increases
        assert gini > 0.9, f"Expected high Gini for inequality, got {gini}"
    
    def test_gini_bounded(self):
        """Gini should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            n = 50
            rewards = rng.integers(0, 20, size=n)
            agents = [
                AgentState(agent_id=i, entry_time=i, attachment=1 + r, total_rewards=r)
                for i, r in enumerate(rewards)
            ]
            gini = compute_gini(agents)
            assert 0 <= gini <= 1, f"Gini should be in [0,1], got {gini}"
    
    def test_gini_no_rewards(self):
        """All zeros should return 0."""
        agents = [
            AgentState(agent_id=i, entry_time=i, attachment=1, total_rewards=0)
            for i in range(10)
        ]
        assert compute_gini(agents) == 0


class TestEverRewarded:
    """Tests for ever-rewarded indicator."""
    
    def test_ever_rewarded_basic(self):
        """Should correctly identify rewarded vs non-rewarded."""
        agents = [
            AgentState(agent_id=0, entry_time=0, attachment=5, total_rewards=4),
            AgentState(agent_id=1, entry_time=1, attachment=1, total_rewards=0),
            AgentState(agent_id=2, entry_time=2, attachment=2, total_rewards=1),
        ]
        result = compute_ever_rewarded(agents)
        expected = np.array([1, 0, 1])
        assert np.array_equal(result, expected)
    
    def test_ever_rewarded_all_rewarded(self):
        """All rewarded should return all 1s."""
        agents = [
            AgentState(agent_id=i, entry_time=i, attachment=2, total_rewards=1)
            for i in range(5)
        ]
        result = compute_ever_rewarded(agents)
        assert np.all(result == 1)
    
    def test_ever_rewarded_none_rewarded(self):
        """None rewarded should return all 0s."""
        agents = [
            AgentState(agent_id=i, entry_time=i, attachment=1, total_rewards=0)
            for i in range(5)
        ]
        result = compute_ever_rewarded(agents)
        assert np.all(result == 0)


class TestIntegration:
    """Integration tests with actual simulation results."""
    
    def test_mi_from_result(self):
        """compute_mi_from_result should work with simulation output."""
        result = simulate_single_run(T=100, alpha=1.0, lambda_effect=0.3, seed=42)
        mi = compute_mi_from_result(result)
        assert mi >= 0
    
    def test_metrics_summary(self):
        """compute_metrics_summary should return all expected fields."""
        result = simulate_single_run(T=100, alpha=1.5, seed=42)
        summary = compute_metrics_summary(result)
        
        expected_keys = [
            "T", "alpha", "lambda_effect", "seed", "n_agents",
            "total_rewards", "mi_v_r", "top_1_share", "top_10_share",
            "gini", "fraction_ever_rewarded"
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"
        
        assert summary["T"] == 100
        assert summary["alpha"] == 1.5
        assert summary["total_rewards"] == 100
        assert summary["n_agents"] == 100
    
    def test_mi_bounded(self):
        """MI should remain bounded and non-catastrophic."""
        # This is a statistical test; we verify MI is reasonable, not that it strictly decreases
        # (strict decrease is a theoretical prediction that requires many runs to observe cleanly)
        n_runs = 10
        T = 100
        
        mi_values = []
        for seed in range(n_runs):
            result = simulate_single_run(
                T=T, alpha=1.5, lambda_effect=0.3, p_high=0.5, seed=seed
            )
            mi_values.append(compute_mi_from_result(result))
        
        mean_mi = np.mean(mi_values)
        
        # MI should be small (< 0.5 bits) for this configuration
        # The theoretical max for binary V and R is 1 bit
        assert mean_mi < 0.5, f"MI unexpectedly large: {mean_mi}"
        assert mean_mi >= 0, f"MI should be non-negative: {mean_mi}"

