"""
Plotting utilities for shadow futures visualization.

Creates publication-quality figures demonstrating:
1. Shadow futures: divergent outcomes from identical work
2. Mutual information collapse as T grows
3. Concentration dynamics under preferential attachment
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from shadow_futures.simulate import ShadowFuturesResult, MIExperimentResult


# Style configuration for distinctive, non-generic aesthetics
STYLE_CONFIG = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.8,
    "legend.facecolor": "#21262d",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
}

# Color palette: cyberpunk-inspired with high contrast
COLORS = {
    "primary": "#58a6ff",      # Electric blue
    "secondary": "#f778ba",    # Hot pink
    "tertiary": "#7ee787",     # Neon green
    "quaternary": "#ffa657",   # Orange
    "quinary": "#a371f7",      # Purple
    "failure": "#f85149",      # Red
    "success": "#3fb950",      # Green
    "neutral": "#8b949e",      # Gray
}


def apply_style() -> None:
    """Apply custom dark style to matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.size"] = 10


def ensure_figures_dir(path: str = "figures") -> Path:
    """Create figures directory if it doesn't exist."""
    fig_path = Path(path)
    fig_path.mkdir(parents=True, exist_ok=True)
    return fig_path


def plot_shadow_futures(
    result: ShadowFuturesResult,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Plot shadow futures demonstration.
    
    Shows distribution of outcomes for a focal agent across simulations
    with identical work transcripts but different random seeds.
    
    Args:
        result: ShadowFuturesResult from experiment
        output_path: Path to save figure (optional)
        show: Whether to display figure
    
    Returns:
        Matplotlib Figure object
    """
    apply_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    
    # Left: Histogram of reward counts
    ax1 = axes[0]
    counts = result.reward_counts
    max_count = max(counts.max(), 1)
    bins = np.arange(-0.5, max_count + 1.5, 1)
    
    ax1.hist(
        counts,
        bins=bins,
        color=COLORS["primary"],
        edgecolor=COLORS["secondary"],
        linewidth=1.5,
        alpha=0.85,
    )
    ax1.axvline(x=0.5, color=COLORS["failure"], linestyle="--", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Reward Count", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title(
        f"Shadow Futures: Agent {result.focal_agent_index}\n"
        f"(α={result.parameters['alpha']}, T={result.parameters['T']})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    
    # Right: Success vs Failure pie chart
    ax2 = axes[1]
    sizes = [result.n_rewarded, result.n_unrewarded]
    labels = [
        f"Rewarded\n({result.n_rewarded}/{result.n_simulations})",
        f"Unrewarded\n({result.n_unrewarded}/{result.n_simulations})",
    ]
    colors = [COLORS["success"], COLORS["failure"]]
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "#c9d1d9", "fontsize": 10},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_color("#0d1117")
        autotext.set_fontweight("bold")
    
    ax2.set_title(
        "Same Work → Different Outcomes\n"
        "(Identical Transcript V)",
        fontsize=12,
        fontweight="bold",
        color="#f0f6fc",
    )
    
    plt.tight_layout()
    
    if output_path:
        fig_dir = ensure_figures_dir(str(Path(output_path).parent))
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    
    if show:
        plt.show()
    
    return fig


def plot_mi_collapse(
    result: MIExperimentResult,
    lambda_idx: int = 0,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Plot mutual information collapse across T values.
    
    For a fixed lambda, shows how MI(V;R) changes with T for different alphas.
    Key prediction: MI decreases with T for alpha >= 1.
    
    Args:
        result: MIExperimentResult from experiment
        lambda_idx: Index of lambda value to plot
        output_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        Matplotlib Figure object
    """
    apply_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")
    
    lam = result.lambdas[lambda_idx]
    color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                   COLORS["quaternary"], COLORS["quinary"]]
    
    # Plot 1: MI vs T for different alphas
    ax1 = axes[0]
    for i, alpha in enumerate(result.alphas):
        mi_vals = result.mi_matrix[:, i, lambda_idx]
        ax1.plot(
            result.T_values,
            mi_vals,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=color_cycle[i % len(color_cycle)],
            label=f"α={alpha}",
        )
    
    ax1.set_xlabel("T (System Size)", fontsize=11)
    ax1.set_ylabel("I(V; R) [bits]", fontsize=11)
    ax1.set_title(f"Mutual Information Collapse\n(λ={lam})", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    
    # Plot 2: Concentration vs T
    ax2 = axes[1]
    for i, alpha in enumerate(result.alphas):
        conc_vals = result.concentration_matrix[:, i, lambda_idx]
        ax2.plot(
            result.T_values,
            conc_vals,
            marker="s",
            markersize=8,
            linewidth=2.5,
            color=color_cycle[i % len(color_cycle)],
            label=f"α={alpha}",
        )
    
    ax2.set_xlabel("T (System Size)", fontsize=11)
    ax2.set_ylabel("Top-1 Share", fontsize=11)
    ax2.set_title("Winner-Take-All Concentration", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    
    # Plot 3: Gini vs T
    ax3 = axes[2]
    for i, alpha in enumerate(result.alphas):
        gini_vals = result.gini_matrix[:, i, lambda_idx]
        ax3.plot(
            result.T_values,
            gini_vals,
            marker="^",
            markersize=8,
            linewidth=2.5,
            color=color_cycle[i % len(color_cycle)],
            label=f"α={alpha}",
        )
    
    ax3.set_xlabel("T (System Size)", fontsize=11)
    ax3.set_ylabel("Gini Coefficient", fontsize=11)
    ax3.set_title("Inequality Growth", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale("log")
    
    plt.tight_layout()
    
    if output_path:
        ensure_figures_dir(str(Path(output_path).parent))
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    
    if show:
        plt.show()
    
    return fig


def plot_mi_heatmap(
    result: MIExperimentResult,
    T_idx: int = -1,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Plot MI as heatmap across alpha and lambda values.
    
    Args:
        result: MIExperimentResult from experiment
        T_idx: Index of T value to plot (-1 = largest)
        output_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        Matplotlib Figure object
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    
    T_val = result.T_values[T_idx]
    mi_data = result.mi_matrix[T_idx, :, :]
    
    im = ax.imshow(
        mi_data,
        aspect="auto",
        cmap="plasma",
        origin="lower",
    )
    
    ax.set_xticks(range(len(result.lambdas)))
    ax.set_xticklabels([f"{l:.2f}" for l in result.lambdas])
    ax.set_yticks(range(len(result.alphas)))
    ax.set_yticklabels([f"{a:.1f}" for a in result.alphas])
    
    ax.set_xlabel("λ (Local Work Effect)", fontsize=11)
    ax.set_ylabel("α (Path Dependence)", fontsize=11)
    ax.set_title(f"I(V; R) at T={T_val}", fontsize=12, fontweight="bold")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mutual Information [bits]", fontsize=10)
    
    # Add value annotations
    for i in range(len(result.alphas)):
        for j in range(len(result.lambdas)):
            val = mi_data[i, j]
            color = "#0d1117" if val > mi_data.max() / 2 else "#f0f6fc"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        ensure_figures_dir(str(Path(output_path).parent))
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    
    if show:
        plt.show()
    
    return fig


def plot_single_run_dynamics(
    T: int = 200,
    alpha: float = 1.5,
    seed: int = 42,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Plot attachment dynamics for a single simulation run.
    
    Shows how early advantages compound under preferential attachment.
    
    Args:
        T: Number of time steps
        alpha: Path dependence exponent
        seed: Random seed
        output_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        Matplotlib Figure object
    """
    from shadow_futures.process import PreferentialAttachmentProcess
    
    apply_style()
    
    process = PreferentialAttachmentProcess(
        T=T,
        alpha=alpha,
        seed=seed,
        track_history=True,
    )
    result = process.run()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    
    # Left: Attachment trajectories over time
    ax1 = axes[0]
    history = result.attachment_history  # 2D array: [time, agent_id]
    
    # Plot trajectories for first 20 agents
    n_show = min(20, len(result.agents))
    for i in range(n_show):
        # Extract attachment for agent i over time (agent enters at t=i)
        entry_time = result.agents[i].entry_time
        traj = history[entry_time:, i]
        if len(traj) > 0:
            times = list(range(entry_time + 1, entry_time + 1 + len(traj)))
            alpha_val = 0.3 + 0.7 * (result.agents[i].total_rewards / max(1, result.T))
            ax1.plot(times, traj, alpha=alpha_val, linewidth=1.5)
    
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("Attachment A(t)", fontsize=11)
    ax1.set_title(f"Path-Dependent Dynamics\n(α={alpha})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Right: Final reward distribution
    ax2 = axes[1]
    rewards = np.array([a.total_rewards for a in result.agents])
    sorted_rewards = np.sort(rewards)[::-1]
    
    ax2.bar(
        range(len(sorted_rewards)),
        sorted_rewards,
        color=COLORS["primary"],
        edgecolor=COLORS["secondary"],
        alpha=0.8,
    )
    ax2.set_xlabel("Agent Rank", fontsize=11)
    ax2.set_ylabel("Total Rewards", fontsize=11)
    ax2.set_title("Final Reward Distribution\n(Winner-Take-All)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        ensure_figures_dir(str(Path(output_path).parent))
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    
    if show:
        plt.show()
    
    return fig

