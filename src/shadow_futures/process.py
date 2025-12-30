"""
Core preferential attachment process for shadow futures demonstration.

This module implements the minimal model from the paper:
- Discrete time t=1..T
- One reward allocated per time step
- Reward probability proportional to A_i(t)^alpha
- Attachment updates: winner gets +1

The model isolates path dependence: identical work transcripts can yield
divergent outcomes based solely on timing and network position.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.random import Generator, PCG64


@dataclass
class AgentState:
    """
    State of a single agent in the preferential attachment process.
    
    Attributes:
        agent_id: Unique identifier for the agent
        entry_time: Time step when the agent entered
        attachment: Current cumulative attachment (reward count + A0)
        transcript: Verified work transcript (for MI experiments)
        total_rewards: Count of rewards received
    """
    agent_id: int
    entry_time: int
    attachment: float
    transcript: int = 0  # 0=low, 1=high for discrete transcript experiments
    total_rewards: int = 0


@dataclass
class SimulationResult:
    """
    Complete result of a single simulation run.
    
    Attributes:
        T: Total time steps
        alpha: Path dependence exponent
        A0: Initial attachment
        lambda_effect: Local work effect weight
        seed: Random seed used
        agents: List of final agent states
        reward_history: Array of winner agent_ids per time step
        attachment_history: Optional array of attachment vectors over time
    """
    T: int
    alpha: float
    A0: float
    lambda_effect: float
    seed: int
    agents: list[AgentState]
    reward_history: np.ndarray
    attachment_history: Optional[np.ndarray] = None


class PreferentialAttachmentProcess:
    """
    Preferential attachment reward allocation process.
    
    Implements the model from Section 2 of the paper:
    - Pr(R_i=1 | A(t)) = A_i(t)^alpha / sum_j A_j(t)^alpha
    - A_i(t+1) = A_i(t) + R_i(t)
    
    With optional local work effect:
    - p = lambda * h(V) + (1-lambda) * f(state)
    where h(V) maps transcript to [0,1] and f(state) is the PA probability.
    
    Parameters:
        T: Number of time steps (also max number of agents if one per step)
        alpha: Path dependence exponent (>=0; higher = stronger reinforcement)
        A0: Initial attachment for all agents on entry
        lambda_effect: Weight of local work effect in [0,1)
        v_low: Transcript value for "low" work agents
        v_high: Transcript value for "high" work agents
        p_high: Probability an agent has high transcript on entry
        entry_rate: Agents entering per time step (default 1)
        seed: Random seed for reproducibility
        track_history: Whether to store full attachment history
    """
    
    def __init__(
        self,
        T: int = 100,
        alpha: float = 1.0,
        A0: float = 1.0,
        lambda_effect: float = 0.0,
        v_low: float = 0.1,
        v_high: float = 0.9,
        p_high: float = 0.5,
        entry_rate: int = 1,
        seed: int = 42,
        track_history: bool = False,
    ):
        self.T = T
        self.alpha = alpha
        self.A0 = A0
        self.lambda_effect = lambda_effect
        self.v_low = v_low
        self.v_high = v_high
        self.p_high = p_high
        self.entry_rate = entry_rate
        self.seed = seed
        self.track_history = track_history
        
        # Initialize random generator with PCG64
        self.rng: Generator = np.random.Generator(PCG64(seed))
        
        # State
        self.agents: list[AgentState] = []
        self.reward_history: list[int] = []
        self.attachment_history: list[np.ndarray] = []
        self.current_time: int = 0
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the process to initial state with optional new seed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.Generator(PCG64(self.seed))
        self.agents = []
        self.reward_history = []
        self.attachment_history = []
        self.current_time = 0
    
    def _add_agent(self) -> AgentState:
        """Add a new agent at current time step."""
        # Assign transcript: 0=low, 1=high
        transcript = 1 if self.rng.random() < self.p_high else 0
        
        agent = AgentState(
            agent_id=len(self.agents),
            entry_time=self.current_time,
            attachment=self.A0,
            transcript=transcript,
            total_rewards=0,
        )
        self.agents.append(agent)
        return agent
    
    def _compute_reward_probs(self) -> np.ndarray:
        """
        Compute reward probability for each agent.
        
        If lambda_effect > 0, combines local work effect with PA:
            p_i = lambda * h(V_i) + (1-lambda) * f(state)
        where h maps transcript to its value and f is the PA probability.
        """
        n = len(self.agents)
        if n == 0:
            return np.array([])
        
        # Get attachments and transcripts
        attachments = np.array([a.attachment for a in self.agents])
        transcripts = np.array([a.transcript for a in self.agents])
        
        # Compute PA probabilities: f(state) = A_i^alpha / sum(A_j^alpha)
        powered = np.power(attachments, self.alpha)
        total = powered.sum()
        if total == 0:
            pa_probs = np.ones(n) / n
        else:
            pa_probs = powered / total
        
        if self.lambda_effect == 0:
            return pa_probs
        
        # Compute work effect: h(V) maps transcript to its value
        work_values = np.where(transcripts == 1, self.v_high, self.v_low)
        work_probs = work_values / work_values.sum() if work_values.sum() > 0 else np.ones(n) / n
        
        # Combine: p = lambda * h(V) + (1-lambda) * f(state)
        combined = self.lambda_effect * work_probs + (1 - self.lambda_effect) * pa_probs
        
        # Normalize to ensure valid probability distribution
        return combined / combined.sum()
    
    def _allocate_reward(self) -> int:
        """Allocate one reward and return winner's agent_id."""
        probs = self._compute_reward_probs()
        if len(probs) == 0:
            return -1
        
        winner_idx = self.rng.choice(len(self.agents), p=probs)
        winner = self.agents[winner_idx]
        winner.attachment += 1
        winner.total_rewards += 1
        return winner.agent_id
    
    def step(self) -> int:
        """
        Execute one time step:
        1. Add new agent(s)
        2. Allocate reward
        3. Update state
        
        Returns:
            Winner's agent_id
        """
        self.current_time += 1
        
        # Add new agents
        for _ in range(self.entry_rate):
            self._add_agent()
        
        # Allocate reward
        winner_id = self._allocate_reward()
        self.reward_history.append(winner_id)
        
        # Track history if requested
        if self.track_history:
            attachments = np.array([a.attachment for a in self.agents])
            self.attachment_history.append(attachments.copy())
        
        return winner_id
    
    def run(self) -> SimulationResult:
        """
        Run the full simulation for T time steps.
        
        Returns:
            SimulationResult with all final states and histories
        """
        self.reset()
        
        for _ in range(self.T):
            self.step()
        
        # Convert attachment history to padded 2D array if tracked
        # Each row is a time step, columns are agents (padded with 0 for not-yet-entered)
        attachment_hist = None
        if self.track_history and self.attachment_history:
            max_agents = len(self.agents)
            attachment_hist = np.zeros((len(self.attachment_history), max_agents))
            for t, attachments in enumerate(self.attachment_history):
                attachment_hist[t, :len(attachments)] = attachments
        
        return SimulationResult(
            T=self.T,
            alpha=self.alpha,
            A0=self.A0,
            lambda_effect=self.lambda_effect,
            seed=self.seed,
            agents=self.agents.copy(),
            reward_history=np.array(self.reward_history),
            attachment_history=attachment_hist,
        )


def simulate_single_run(
    T: int = 100,
    alpha: float = 1.0,
    A0: float = 1.0,
    lambda_effect: float = 0.0,
    v_low: float = 0.1,
    v_high: float = 0.9,
    p_high: float = 0.5,
    seed: int = 42,
    track_history: bool = False,
) -> SimulationResult:
    """
    Convenience function to run a single simulation.
    
    Args:
        T: Number of time steps
        alpha: Path dependence exponent
        A0: Initial attachment
        lambda_effect: Local work effect weight
        v_low: Low transcript value
        v_high: High transcript value
        p_high: Probability of high transcript
        seed: Random seed
        track_history: Whether to track full attachment history
    
    Returns:
        SimulationResult with final states
    """
    process = PreferentialAttachmentProcess(
        T=T,
        alpha=alpha,
        A0=A0,
        lambda_effect=lambda_effect,
        v_low=v_low,
        v_high=v_high,
        p_high=p_high,
        seed=seed,
        track_history=track_history,
    )
    return process.run()

