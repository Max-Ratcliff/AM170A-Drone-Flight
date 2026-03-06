"""
Statistical benchmarking for drone routing solvers.
Quantifies the optimality gap and heuristic gain across varying N.
"""

import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from optimizer import RoutingOptimizer
from params import SimulationConfig, get_default_params
from physics import DronePhysics
from targets import Targets

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def run_benchmark(n_range: range, trials_per_n: int = 10):
    """
    Runs benchmarks for N in n_range, with specified trials per N.
    Returns a dictionary of results.
    """
    results = {
        "n": [],
        "gap_mean": [],
        "gap_std": [],
        "gain_mean": [],
        "gain_std": [],
        "times_hk": [],
        "times_2opt": [],
    }

    params = get_default_params()
    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)

    for n in n_range:
        print(f"Benchmarking N={n}...")
        gaps = []
        gains = []
        times_hk = []
        times_2opt = []

        for seed in range(trials_per_n):
            targets = Targets(num_targets=n, bounds=(0, 2000), seed=seed)
            waypoints = targets.generate_waypoints()

            energy_matrix, _ = optimizer.build_energy_matrix(waypoints)

            # Held-Karp (Global Optimum)
            start = time.time()
            hk_order = optimizer.solve_tsp(energy_matrix, method="held_karp")
            times_hk.append(time.time() - start)
            e_min = optimizer._tour_cost(energy_matrix, hk_order)

            # Nearest Neighbor
            nn_order = optimizer.solve_tsp(energy_matrix, method="nearest_neighbor")
            e_nn = optimizer._tour_cost(energy_matrix, nn_order)

            # NN + 2-Opt
            start = time.time()
            opt2_order = optimizer.solve_tsp(energy_matrix, method="nn_2opt")
            times_2opt.append(time.time() - start)
            e_2opt = optimizer._tour_cost(energy_matrix, opt2_order)

            # Ratios
            gaps.append(e_2opt / e_min)
            gains.append(e_2opt / e_nn)

        results["n"].append(n)
        results["gap_mean"].append(np.mean(gaps))
        results["gap_std"].append(np.std(gaps))
        results["gain_mean"].append(np.mean(gains))
        results["gain_std"].append(np.std(gains))
        results["times_hk"].append(np.mean(times_hk))
        results["times_2opt"].append(np.mean(times_2opt))

    return results


def plot_benchmark_results(results: Dict):
    """Generates the benchmark ratio plot for the paper."""
    fig, ax = plt.subplots(figsize=(12, 8))

    n = results["n"]

    # Plot Gap (2-opt / Held-Karp)
    ax.errorbar(
        n,
        results["gap_mean"],
        yerr=results["gap_std"],
        fmt="o-",
        color="blue",
        capsize=5,
        label="Optimality Gap ($E_{2opt}/E_{min}$)",
    )

    # Plot Gain (2-opt / NN)
    ax.errorbar(
        n,
        results["gain_mean"],
        yerr=results["gain_std"],
        fmt="s--",
        color="green",
        capsize=5,
        label="Heuristic Gain ($E_{2opt}/E_{NN}$)",
    )

    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5)

    ax.set_xlabel("Number of Waypoints ($N$)", fontsize=18)
    ax.set_ylabel("Energy Ratio", fontsize=18)
    ax.set_title("Solver Performance Benchmarks", fontsize=22)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.xticks(n)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "benchmark_ratios.png", dpi=300)
    print(f"Benchmark plot saved to {PLOTS_DIR / 'benchmark_ratios.png'}")
    plt.close(fig)


if __name__ == "__main__":
    # N from 4 to 12 (Held-Karp gets slow beyond 15)
    bench_results = run_benchmark(range(3, 20), trials_per_n=10)
    plot_benchmark_results(bench_results)
