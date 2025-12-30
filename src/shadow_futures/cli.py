"""
Command-line interface for shadow futures experiments.

Provides three main commands:
1. simulate: Run simulations and output summary JSON or CSV
2. plot-mi: Generate mutual information collapse plots and CSV
3. shadow-futures: Demonstrate shadow futures concept with CSV export
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from shadow_futures.simulate import (
    run_single_simulation_summary,
    run_shadow_futures_experiment,
    run_mi_experiment,
    simulate_single_run,
)
from shadow_futures.metrics import compute_metrics_summary
from shadow_futures.plots import (
    plot_shadow_futures,
    plot_mi_collapse,
    plot_mi_heatmap,
    plot_single_run_dynamics,
    ensure_figures_dir,
)


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run simulation and output summary."""
    summary = run_single_simulation_summary(
        T=args.T,
        alpha=args.alpha,
        A0=args.A0,
        lambda_effect=args.lambda_effect,
        p_high=args.p_high,
        seed=args.seed,
    )
    
    # Also get per-agent data if CSV requested
    if args.csv:
        result = simulate_single_run(
            T=args.T,
            alpha=args.alpha,
            A0=args.A0,
            lambda_effect=args.lambda_effect,
            p_high=args.p_high,
            seed=args.seed,
        )
        
        fig_dir = ensure_figures_dir(args.output_dir)
        
        # Save summary CSV
        summary_path = fig_dir / "simulation_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in summary.items():
                writer.writerow([key, value])
        print(f"Summary CSV saved to: {summary_path}")
        
        # Save per-agent CSV
        agents_path = fig_dir / "simulation_agents.csv"
        with open(agents_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "entry_time", "transcript", "total_rewards", "attachment"])
            for agent in result.agents:
                writer.writerow([
                    agent.agent_id,
                    agent.entry_time,
                    agent.transcript,
                    agent.total_rewards,
                    agent.attachment,
                ])
        print(f"Per-agent CSV saved to: {agents_path}")
        
        # Save reward history CSV
        history_path = fig_dir / "simulation_reward_history.csv"
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_step", "winner_agent_id"])
            for t, winner in enumerate(result.reward_history):
                writer.writerow([t + 1, winner])
        print(f"Reward history CSV saved to: {history_path}")
    
    # Output JSON to file or stdout
    if args.output:
        ensure_figures_dir(str(Path(args.output).parent) or ".")
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON written to {args.output}")
    elif not args.csv:
        print(json.dumps(summary, indent=2))
    
    return 0


def cmd_plot_mi(args: argparse.Namespace) -> int:
    """Generate mutual information collapse plots."""
    print("Running MI experiment...")
    print(f"  T values: {args.T_values}")
    print(f"  alphas: {args.alphas}")
    print(f"  lambdas: {args.lambdas}")
    print(f"  runs per point: {args.n_runs}")
    
    result = run_mi_experiment(
        T_values=args.T_values,
        alphas=args.alphas,
        lambdas=args.lambdas,
        n_runs=args.n_runs,
        base_seed=args.seed,
        A0=args.A0,
        p_high=args.p_high,
    )
    
    fig_dir = ensure_figures_dir(args.output_dir)
    
    # Generate plots for each lambda value
    for i, lam in enumerate(result.lambdas):
        output_path = fig_dir / f"mi_collapse_lambda_{lam:.2f}.png"
        plot_mi_collapse(result, lambda_idx=i, output_path=str(output_path))
        print(f"  Saved: {output_path}")
    
    # Generate heatmap for largest T
    heatmap_path = fig_dir / "mi_heatmap.png"
    plot_mi_heatmap(result, T_idx=-1, output_path=str(heatmap_path))
    print(f"  Saved: {heatmap_path}")
    
    # Save CSV if requested
    if args.save_csv:
        # Save MI results as CSV (long format for easy analysis)
        mi_csv_path = fig_dir / "mi_experiment_results.csv"
        with open(mi_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["T", "alpha", "lambda", "mi_mean", "mi_std", "top1_mean", "top1_std", "gini_mean", "gini_std", "n_runs"])
            for i_T, T in enumerate(result.T_values):
                for i_a, alpha in enumerate(result.alphas):
                    for i_l, lam in enumerate(result.lambdas):
                        writer.writerow([
                            T,
                            alpha,
                            lam,
                            result.mi_matrix[i_T, i_a, i_l],
                            result.mi_std_matrix[i_T, i_a, i_l],
                            result.concentration_matrix[i_T, i_a, i_l],
                            result.concentration_std_matrix[i_T, i_a, i_l],
                            result.gini_matrix[i_T, i_a, i_l],
                            result.gini_std_matrix[i_T, i_a, i_l],
                            result.n_runs_per_point,
                        ])
        print(f"  CSV saved: {mi_csv_path}")
    
    # Print summary statistics with std
    print("\n=== Mutual Information Summary ===")
    print(f"R = 'ever rewarded by time T' (binary). MI in bits (log2).")
    print(f"Note: When lambda=0, MI should be ~0; positive values reflect estimator bias.")
    print(f"\nLargest T = {result.T_values[-1]} (n_runs={result.n_runs_per_point}):")
    for i, alpha in enumerate(result.alphas):
        for j, lam in enumerate(result.lambdas):
            mi = result.mi_matrix[-1, i, j]
            mi_std = result.mi_std_matrix[-1, i, j]
            conc = result.concentration_matrix[-1, i, j]
            conc_std = result.concentration_std_matrix[-1, i, j]
            print(f"  alpha={alpha:.1f}, lambda={lam:.2f}: I(V;R)={mi:.4f}+/-{mi_std:.4f} bits, Top-1={conc:.3f}+/-{conc_std:.3f}")
    
    return 0


def cmd_shadow_futures(args: argparse.Namespace) -> int:
    """Demonstrate shadow futures concept."""
    print("Running shadow futures experiment...")
    print(f"  Focal agent: {args.focal_agent}")
    print(f"  T={args.T}, alpha={args.alpha}")
    print(f"  {args.n_simulations} simulations")
    
    result = run_shadow_futures_experiment(
        T=args.T,
        alpha=args.alpha,
        A0=args.A0,
        focal_agent_index=args.focal_agent,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        lambda_effect=args.lambda_effect,
    )
    
    fig_dir = ensure_figures_dir(args.output_dir)
    output_path = fig_dir / "shadow_futures.png"
    plot_shadow_futures(result, output_path=str(output_path))
    
    # Save CSV if requested (with parameter-based filenames to prevent overwriting)
    if args.save_csv:
        param_suffix = f"_T{args.T}_a{args.alpha}_agent{args.focal_agent}_n{args.n_simulations}"
        
        # Save per-simulation results
        sim_csv_path = fig_dir / f"shadow_futures_simulations{param_suffix}.csv"
        with open(sim_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["simulation_id", "seed", "focal_agent_rewards", "focal_agent_rewarded"])
            base_seed = result.parameters["base_seed"]
            for i, reward_count in enumerate(result.reward_counts):
                writer.writerow([
                    i,
                    base_seed + i,
                    reward_count,
                    1 if reward_count > 0 else 0,
                ])
        print(f"  Per-simulation CSV saved: {sim_csv_path}")
        
        # Save summary statistics
        summary_csv_path = fig_dir / f"shadow_futures_summary{param_suffix}.csv"
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["focal_agent_index", result.focal_agent_index])
            writer.writerow(["transcript", result.transcript])
            writer.writerow(["n_simulations", result.n_simulations])
            writer.writerow(["n_rewarded", result.n_rewarded])
            writer.writerow(["n_unrewarded", result.n_unrewarded])
            writer.writerow(["pct_rewarded", 100 * result.n_rewarded / result.n_simulations])
            writer.writerow(["reward_mean", result.reward_counts.mean()])
            writer.writerow(["reward_std", result.reward_counts.std()])
            writer.writerow(["reward_min", result.reward_counts.min()])
            writer.writerow(["reward_max", result.reward_counts.max()])
            writer.writerow(["reward_median", np.median(result.reward_counts)])
            writer.writerow(["T", result.parameters["T"]])
            writer.writerow(["alpha", result.parameters["alpha"]])
            writer.writerow(["A0", result.parameters["A0"]])
            writer.writerow(["lambda_effect", result.parameters["lambda_effect"]])
        print(f"  Summary CSV saved: {summary_csv_path}")
    
    # Print textual summary
    print("\n" + "=" * 50)
    print("SHADOW FUTURES DEMONSTRATION")
    print("=" * 50)
    print(f"\nFocal Agent {result.focal_agent_index}:")
    print(f"  Work transcript V = {result.transcript} (identical across all simulations)")
    print(f"\nOutcomes across {result.n_simulations} simulations:")
    print(f"  Rewarded:   {result.n_rewarded:4d} ({100*result.n_rewarded/result.n_simulations:.1f}%)")
    print(f"  Unrewarded: {result.n_unrewarded:4d} ({100*result.n_unrewarded/result.n_simulations:.1f}%)")
    print(f"\nReward count statistics:")
    print(f"  Mean:   {result.reward_counts.mean():.2f}")
    print(f"  Std:    {result.reward_counts.std():.2f}")
    print(f"  Min:    {result.reward_counts.min()}")
    print(f"  Max:    {result.reward_counts.max()}")
    print(f"  Median: {np.median(result.reward_counts):.0f}")
    
    print("\n" + "-" * 50)
    print("INTERPRETATION")
    print("-" * 50)
    print("""
Despite IDENTICAL verified work (same transcript V), the focal agent
experiences radically different outcomes across simulation runs.

This demonstrates SHADOW FUTURES: unrealized trajectories with identical
work that fail due solely to unfavorable timing or network position.

What varies across seeds is the realized reward history, hence the
allocation state S(t). The focal agent's work is constant; only the
path-dependent state differs. Divergent outcomes are structural, not
a matter of 'luck' in any moral sense.

Key insight: Verification confirms work occurred, but cannot establish
that work caused the outcome. Observed success reflects realized position.
""")
    
    print(f"\nFigure saved to: {output_path}")
    
    # Also generate dynamics plot
    dynamics_path = fig_dir / "dynamics.png"
    plot_single_run_dynamics(
        T=args.T,
        alpha=args.alpha,
        seed=args.seed,
        output_path=str(dynamics_path),
    )
    print(f"Dynamics plot saved to: {dynamics_path}")
    
    return 0


def parse_list(s: str, dtype=float) -> list:
    """Parse comma-separated string into list."""
    return [dtype(x.strip()) for x in s.split(",")]


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="shadow-futures",
        description="Shadow Futures: Demonstrating path-dependent allocation dynamics",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # === simulate command ===
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Run single simulation and output summary",
    )
    sim_parser.add_argument("--T", type=int, default=100, help="Time steps (default: 100)")
    sim_parser.add_argument("--alpha", type=float, default=1.0, help="Path dependence exponent (default: 1.0)")
    sim_parser.add_argument("--A0", type=float, default=1.0, help="Initial attachment (default: 1.0)")
    sim_parser.add_argument("--lambda-effect", type=float, default=0.0, help="Local work effect (default: 0.0)")
    sim_parser.add_argument("--p-high", type=float, default=0.5, help="Prob of high transcript (default: 0.5)")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    sim_parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    sim_parser.add_argument("--csv", action="store_true", help="Save detailed CSV files to output-dir")
    sim_parser.add_argument("--output-dir", type=str, default="figures", help="Output directory for CSV (default: figures)")
    sim_parser.set_defaults(func=cmd_simulate)
    
    # === plot-mi command ===
    mi_parser = subparsers.add_parser(
        "plot-mi",
        help="Generate mutual information collapse plots",
    )
    mi_parser.add_argument(
        "--T-values",
        type=lambda s: parse_list(s, int),
        default=[50, 100, 200, 500],
        help="Comma-separated T values (default: 50,100,200,500)",
    )
    mi_parser.add_argument(
        "--alphas",
        type=lambda s: parse_list(s, float),
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Comma-separated alpha values (default: 0.0,0.5,1.0,1.5,2.0)",
    )
    mi_parser.add_argument(
        "--lambdas",
        type=lambda s: parse_list(s, float),
        default=[0.0, 0.1, 0.3],
        help="Comma-separated lambda values (default: 0.0,0.1,0.3)",
    )
    mi_parser.add_argument("--n-runs", type=int, default=50, help="Runs per data point (default: 50)")
    mi_parser.add_argument("--A0", type=float, default=1.0, help="Initial attachment (default: 1.0)")
    mi_parser.add_argument("--p-high", type=float, default=0.5, help="Prob of high transcript (default: 0.5)")
    mi_parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    mi_parser.add_argument("--output-dir", type=str, default="figures", help="Output directory (default: figures)")
    mi_parser.add_argument("--save-csv", action="store_true", help="Save results to CSV file")
    mi_parser.set_defaults(func=cmd_plot_mi)
    
    # === shadow-futures command ===
    sf_parser = subparsers.add_parser(
        "shadow-futures",
        help="Demonstrate shadow futures concept",
    )
    sf_parser.add_argument("--T", type=int, default=100, help="Time steps (default: 100)")
    sf_parser.add_argument("--alpha", type=float, default=1.5, help="Path dependence exponent (default: 1.5)")
    sf_parser.add_argument("--A0", type=float, default=1.0, help="Initial attachment (default: 1.0)")
    sf_parser.add_argument("--focal-agent", type=int, default=10, help="Focal agent index (default: 10)")
    sf_parser.add_argument("--n-simulations", type=int, default=500, help="Number of simulations (default: 500)")
    sf_parser.add_argument("--lambda-effect", type=float, default=0.0, help="Local work effect (default: 0.0)")
    sf_parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    sf_parser.add_argument("--output-dir", type=str, default="figures", help="Output directory (default: figures)")
    sf_parser.add_argument("--save-csv", action="store_true", help="Save results to CSV files")
    sf_parser.set_defaults(func=cmd_shadow_futures)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
