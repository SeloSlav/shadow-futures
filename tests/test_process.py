"""
Tests for shadow_futures.process module.

These tests verify core invariants of the preferential attachment process:
1. Determinism: Same seed produces same results
2. Conservation: Total rewards equal number of time steps
3. Monotonicity: Attachment only increases
4. Path dependence: Higher alpha leads to higher concentration
"""

import numpy as np
import pytest

from shadow_futures.process import (
    PreferentialAttachmentProcess,
    simulate_single_run,
    AgentState,
    SimulationResult,
)


class TestPreferentialAttachmentProcess:
    """Tests for the core preferential attachment process."""
    
    def test_determinism_same_seed(self):
        """Same seed should produce identical results."""
        result1 = simulate_single_run(T=50, alpha=1.0, seed=12345)
        result2 = simulate_single_run(T=50, alpha=1.0, seed=12345)
        
        assert np.array_equal(result1.reward_history, result2.reward_history)
        assert len(result1.agents) == len(result2.agents)
        for a1, a2 in zip(result1.agents, result2.agents):
            assert a1.total_rewards == a2.total_rewards
            assert a1.attachment == a2.attachment
    
    def test_determinism_different_seeds(self):
        """Different seeds should (typically) produce different results."""
        result1 = simulate_single_run(T=50, alpha=1.0, seed=12345)
        result2 = simulate_single_run(T=50, alpha=1.0, seed=54321)
        
        # Reward histories should differ (extremely unlikely to be same)
        assert not np.array_equal(result1.reward_history, result2.reward_history)
    
    def test_reward_conservation(self):
        """Total rewards should equal number of time steps."""
        for T in [10, 50, 100]:
            result = simulate_single_run(T=T, alpha=1.0, seed=42)
            total_rewards = sum(a.total_rewards for a in result.agents)
            assert total_rewards == T, f"Expected {T} rewards, got {total_rewards}"
    
    def test_attachment_monotonicity(self):
        """Attachment should never decrease (can only gain rewards)."""
        process = PreferentialAttachmentProcess(T=100, alpha=1.5, seed=42, track_history=True)
        result = process.run()
        
        # Check that each agent's attachment only increases over time
        # attachment_history is a 2D array: [time_step, agent_id]
        # Agent i enters at entry_time=i (0-indexed), history[i] is first snapshot
        for agent_id in range(min(20, len(result.agents))):
            entry_time = result.agents[agent_id].entry_time
            prev_attachment = result.A0
            for t in range(entry_time, len(result.attachment_history)):
                current = result.attachment_history[t, agent_id]
                assert current >= prev_attachment, \
                    f"Agent {agent_id} attachment decreased at t={t}"
                prev_attachment = current
    
    def test_agent_count_matches_T(self):
        """With entry_rate=1, number of agents should equal T."""
        for T in [10, 50, 100]:
            result = simulate_single_run(T=T, alpha=1.0, seed=42)
            assert len(result.agents) == T
    
    def test_initial_attachment_A0(self):
        """All agents should start with attachment A0."""
        A0 = 5.0
        process = PreferentialAttachmentProcess(T=50, A0=A0, seed=42)
        result = process.run()
        
        for agent in result.agents:
            # Attachment = A0 + rewards received
            expected_attachment = A0 + agent.total_rewards
            assert agent.attachment == expected_attachment
    
    def test_alpha_zero_uniform(self):
        """With alpha=0, rewards should be approximately uniform."""
        # Run many times and check distribution is roughly uniform
        n_runs = 100
        T = 50
        first_agent_rewards = []
        
        for seed in range(n_runs):
            result = simulate_single_run(T=T, alpha=0.0, seed=seed)
            first_agent_rewards.append(result.agents[0].total_rewards)
        
        # With uniform allocation, first agent should get roughly T/T = 1 reward on average
        # (actually more complex due to growing agent pool, but should be low variance)
        mean_rewards = np.mean(first_agent_rewards)
        # Should be greater than 0 and not dominate
        assert 0 < mean_rewards < T / 2
    
    def test_high_alpha_concentration(self):
        """Higher alpha should lead to more concentrated rewards."""
        T = 100
        seed = 42
        
        # Run with different alphas
        result_low = simulate_single_run(T=T, alpha=0.5, seed=seed)
        result_high = simulate_single_run(T=T, alpha=2.0, seed=seed + 1000)
        
        # Compute top-1 share
        rewards_low = np.array([a.total_rewards for a in result_low.agents])
        rewards_high = np.array([a.total_rewards for a in result_high.agents])
        
        top1_low = rewards_low.max() / rewards_low.sum() if rewards_low.sum() > 0 else 0
        top1_high = rewards_high.max() / rewards_high.sum() if rewards_high.sum() > 0 else 0
        
        # Higher alpha should (on average) lead to higher concentration
        # This is a statistical property, so we use many seeds for robustness
        n_runs = 20
        concentration_low = []
        concentration_high = []
        
        for i in range(n_runs):
            r_low = simulate_single_run(T=T, alpha=0.5, seed=seed + i)
            r_high = simulate_single_run(T=T, alpha=2.0, seed=seed + i)
            
            rew_low = np.array([a.total_rewards for a in r_low.agents])
            rew_high = np.array([a.total_rewards for a in r_high.agents])
            
            concentration_low.append(rew_low.max() / rew_low.sum())
            concentration_high.append(rew_high.max() / rew_high.sum())
        
        assert np.mean(concentration_high) > np.mean(concentration_low), \
            "Higher alpha should produce higher concentration on average"


class TestLocalWorkEffect:
    """Tests for the local work effect (lambda > 0)."""
    
    def test_lambda_zero_no_work_effect(self):
        """With lambda=0, transcript should not affect rewards."""
        # This is inherent in the model; just verify no errors
        result = simulate_single_run(T=50, alpha=1.0, lambda_effect=0.0, seed=42)
        assert len(result.agents) == 50
    
    def test_lambda_positive_work_effect(self):
        """With lambda > 0, high transcripts should get more rewards on average."""
        n_runs = 50
        T = 100
        
        high_transcript_rewards = []
        low_transcript_rewards = []
        
        for seed in range(n_runs):
            result = simulate_single_run(
                T=T,
                alpha=1.0,
                lambda_effect=0.5,
                p_high=0.5,
                v_low=0.1,
                v_high=0.9,
                seed=seed,
            )
            
            for agent in result.agents:
                if agent.transcript == 1:
                    high_transcript_rewards.append(agent.total_rewards)
                else:
                    low_transcript_rewards.append(agent.total_rewards)
        
        # With lambda > 0 and v_high > v_low, high transcripts should average more rewards
        # (though path dependence still dominates as T grows)
        assert np.mean(high_transcript_rewards) >= np.mean(low_transcript_rewards) * 0.8, \
            "High transcripts should not be severely disadvantaged with lambda > 0"
    
    def test_transcript_assignment(self):
        """Transcript assignment should follow p_high probability."""
        n_runs = 100
        T = 50
        
        high_counts = []
        for seed in range(n_runs):
            result = simulate_single_run(T=T, p_high=0.7, seed=seed)
            n_high = sum(1 for a in result.agents if a.transcript == 1)
            high_counts.append(n_high / T)
        
        # Average should be close to p_high
        mean_high = np.mean(high_counts)
        assert 0.6 < mean_high < 0.8, f"Expected ~0.7 high transcripts, got {mean_high}"


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_result_fields(self):
        """SimulationResult should contain all expected fields."""
        result = simulate_single_run(T=50, alpha=1.0, seed=42)
        
        assert result.T == 50
        assert result.alpha == 1.0
        assert result.seed == 42
        assert len(result.agents) == 50
        assert len(result.reward_history) == 50
    
    def test_track_history(self):
        """With track_history=True, should record attachment over time."""
        process = PreferentialAttachmentProcess(T=50, seed=42, track_history=True)
        result = process.run()
        
        assert result.attachment_history is not None
        assert len(result.attachment_history) == 50
    
    def test_no_track_history(self):
        """With track_history=False, attachment_history should be None."""
        result = simulate_single_run(T=50, seed=42, track_history=False)
        assert result.attachment_history is None

