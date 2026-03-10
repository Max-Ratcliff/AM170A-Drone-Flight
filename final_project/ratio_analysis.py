"""
Ratio analysis: E_2opt/E_NN and E_2opt/E_min as a function of N.

For each N, generates num_configs random waypoint configurations,
computes route energies using brute force (or held_karp), NN, and NN+2opt,
then averages the ratios across configurations and plots vs N.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from optimizer import RoutingOptimizer
from params import SimulationConfig, get_default_params
from physics import DronePhysics
from targets import Targets

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def route_energy(order: list[int], energy_matrix: np.ndarray) -> float:
    total = 0.0
    for k in range(len(order)):
        i, j = order[k], order[(k + 1) % len(order)]
        total += energy_matrix[i, j]
    return total


def run_analysis(
    n_values: list[int],
    num_configs: int = 10,
    brute_cutoff: int = 9,
    held_karp_cutoff: int = 15,
) -> None:
    params = get_default_params()
    physics = DronePhysics(params)
    optimizer = RoutingOptimizer(physics)

    mean_ratio_2opt_nn = []
    std_ratio_2opt_nn = []
    mean_ratio_2opt_min = []
    std_ratio_2opt_min = []
    n_with_exact = []

    for n in n_values:
        ratios_2opt_nn = []
        ratios_2opt_min = []
        print(f"N = {n} ...", end=" ", flush=True)

        for trial in range(num_configs):
            config = SimulationConfig(
                num_targets=n,
                bounds=(0.0, 2000.0),
                seed=1000 * n + trial,
            )
            targets = Targets(
                num_targets=config.num_targets,
                bounds=config.bounds,
                seed=config.seed,
            )
            waypoints = targets.generate_waypoints()
            energy_matrix, _ = optimizer.build_energy_matrix(waypoints)

            # NN
            nn_order = optimizer.solve_tsp(energy_matrix, method="nearest_neighbor")
            e_nn = route_energy(nn_order, energy_matrix)

            # NN + 2-opt
            opt2_order = optimizer.solve_tsp(energy_matrix, method="nn_2opt")
            e_2opt = route_energy(opt2_order, energy_matrix)

            ratios_2opt_nn.append(e_2opt / e_nn)

            # Exact solution for small N
            if n <= brute_cutoff:
                exact_order = optimizer.solve_tsp(energy_matrix, method="brute")
                e_min = route_energy(exact_order, energy_matrix)
                ratios_2opt_min.append(e_2opt / e_min)
            elif n <= held_karp_cutoff:
                exact_order = optimizer.solve_tsp(energy_matrix, method="held_karp")
                e_min = route_energy(exact_order, energy_matrix)
                ratios_2opt_min.append(e_2opt / e_min)

        mean_ratio_2opt_nn.append(np.mean(ratios_2opt_nn))
        std_ratio_2opt_nn.append(np.std(ratios_2opt_nn))

        if ratios_2opt_min:
            n_with_exact.append(n)
            mean_ratio_2opt_min.append(np.mean(ratios_2opt_min))
            std_ratio_2opt_min.append(np.std(ratios_2opt_min))

        print(f"E2/ENN = {np.mean(ratios_2opt_nn):.4f}", end="")
        if ratios_2opt_min:
            print(f", E2/Emin = {np.mean(ratios_2opt_min):.4f}", end="")
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(
        n_values,
        mean_ratio_2opt_nn,
        yerr=std_ratio_2opt_nn,
        fmt="o-",
        color="blue",
        capsize=5,
        linewidth=2,
        markersize=8,
        label=r"$E_{\mathrm{2opt}} / E_{\mathrm{NN}}$",
    )

    if n_with_exact:
        ax.errorbar(
            n_with_exact,
            mean_ratio_2opt_min,
            yerr=std_ratio_2opt_min,
            fmt="s--",
            color="red",
            capsize=5,
            linewidth=2,
            markersize=8,
            label=r"$E_{\mathrm{2opt}} / E_{\min}$",
        )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Number of waypoints $N$", fontsize=16)
    ax.set_ylabel("Energy ratio (averaged over 10 configurations)", fontsize=16)
    ax.set_title(
        "Algorithm performance: NN+2opt improvement and optimality gap",
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "ratio_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    # Brute force up to 9, Held-Karp up to 15, NN+2opt vs NN for all
    n_vals = [4, 5, 6, 7, 8, 9, 12, 15, 20, 30, 50]
    run_analysis(n_vals, num_configs=10, brute_cutoff=9, held_karp_cutoff=15)
