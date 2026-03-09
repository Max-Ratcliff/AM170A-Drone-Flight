"""
Statistical benchmarking for drone routing solvers.
Quantifies the optimality gap and heuristic gain across varying N (3 to 20).
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from optimizer import RoutingOptimizer
from params import get_default_params
from physics import DronePhysics
from plotting import save_plot
from targets import Targets

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def run_benchmark(n_range: range, trials_per_n: int = 10):
    """
    Runs benchmarks for N in n_range.
    For each N, generates multiple configurations and computes:
    - E_min (Held-Karp)
    - E_NN (Nearest Neighbor)
    - E_2opt (NN + 2-opt)
    """
    results = {
        "n": [],
        "gap_mean": [],
        "gain_mean": [],
    }

    params = get_default_params()
    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)

    for n in n_range:
        print(f"Benchmarking N={n}...")
        gaps = []
        gains = []

        for seed in range(trials_per_n):
            targets = Targets(num_targets=n, bounds=(0, 2000), seed=seed)
            waypoints = targets.generate_waypoints()
            energy_matrix, _ = optimizer.build_energy_matrix(waypoints)

            # Global Optimum (Held-Karp)
            hk_order = optimizer.solve_tsp(energy_matrix, method="held_karp")
            e_min = optimizer._tour_cost(energy_matrix, hk_order)

            # Nearest Neighbor
            nn_order = optimizer.solve_tsp(energy_matrix, method="nearest_neighbor")
            e_nn = optimizer._tour_cost(energy_matrix, nn_order)

            # NN + 2-Opt
            opt2_order = optimizer.solve_tsp(energy_matrix, method="nn_2opt")
            e_2opt = optimizer._tour_cost(energy_matrix, opt2_order)

            # Calculate Ratios
            gaps.append(e_2opt / e_min)
            gains.append(e_2opt / e_nn)

        results["n"].append(n)
        results["gap_mean"].append(np.mean(gaps))
        results["gain_mean"].append(np.mean(gains))

    return results


def plot_benchmark_results(results: Dict):
    """Generates the benchmark ratio plot."""
    fig, ax = plt.subplots(figsize=(12, 8))

    n = results["n"]

    # Plot Gap (2-opt / Held-Karp)
    ax.plot(
        n,
        results["gap_mean"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=8,
        label=r"Optimality Gap ($\bar{E}_{2opt}/\bar{E}_{min}$)",
    )

    # Plot Gain (2-opt / NN)
    ax.plot(
        n,
        results["gain_mean"],
        "s--",
        color="green",
        linewidth=2,
        markersize=8,
        label=r"Heuristic Gain ($\bar{E}_{2opt}/\bar{E}_{NN}$)",
    )

    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5)

    ax.set_xlabel("Number of Waypoints ($N$)", fontsize=18)
    ax.set_ylabel("Mean Energy Ratio", fontsize=18)
    ax.set_title("Solver Performance Benchmarks (N=3 to 20)", fontsize=22)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.xticks(n)
    fig.tight_layout()

    save_plot(fig, "benchmark_ratios.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # N from 3 to 20 as requested
    bench_results = run_benchmark(range(3, 21), trials_per_n=10)
    plot_benchmark_results(bench_results)
