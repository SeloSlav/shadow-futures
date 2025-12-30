"""
Command-line interface for shadow futures experiments.

Provides three main commands:
1. simulate: Run simulations and output summary JSON
2. plot-mi: Generate mutual information collapse plots
3. shadow-futures: Demonstrate shadow futures concept
"""

import argparse
import json
import sys
from pathlib import Path

from shadow_futures.simulate import (
    run_single_simulation_summary,
    run_shadow_futures_experiment,
    run_mi_experiment,
)
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
    
    # Output to file or stdout
    if args.output:
        ensure_figures_dir(str(Path(args.output).parent) or ".")
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.output}")
    else:
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
    
    # Print summary statistics
    print("\n=== Mutual Information Summary ===")
    print(f"Largest T = {result.T_values[-1]}:")
    for i, alpha in enumerate(result.alphas):
        for j, lam in enumerate(result.lambdas):
            mi = result.mi_matrix[-1, i, j]
            conc = result.concentration_matrix[-1, i, j]
            print(f"  alpha={alpha:.1f}, lambda={lam:.2f}: I(V;R)={mi:.4f} bits, Top-1={conc:.3f}")
    
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
    print(f"  Median: {int(result.reward_counts[len(result.reward_counts)//2])}")
    
    print("\n" + "-" * 50)
    print("INTERPRETATION")
    print("-" * 50)
    print("""
Despite IDENTICAL verified work (same transcript V), the focal agent
experiences radically different outcomes across simulation runs.

This demonstrates SHADOW FUTURES: unrealized trajectories with identical
work that fail due solely to unfavorable timing or network position.

The variation in outcomes cannot be attributed to differences in effort
or work quality - it is purely a consequence of path-dependent allocation.

Key insight: Verification confirms work occurred, but cannot establish
that work caused the outcome. Success is evidence of position, not merit.
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
    sf_parser.set_defaults(func=cmd_shadow_futures)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

