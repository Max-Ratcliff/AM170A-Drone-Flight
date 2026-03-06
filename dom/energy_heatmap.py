"""
Energy and Velocity Heatmap Visualization

Plots the N×N energy cost matrix and N×N optimal velocity matrix as heatmaps.
Asymmetry in the energy heatmap reveals wind effects (i→j ≠ j→i).

Usage:
    python dom/energy_heatmap.py                  # 5 random waypoints, no wind
    python dom/energy_heatmap.py --test           # fixed test set with wind
    python dom/energy_heatmap.py -n 6 -s 42 -w 5.0 -2.0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final_project"))

import argparse

import matplotlib.pyplot as plt
import numpy as np

from optimizer import RoutingOptimizer
from params import (SimulationConfig, get_default_params,
                    get_default_sim_config, get_test_sim_config)
from physics import DronePhysics
from targets import Targets

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def plot_heatmap(matrix, labels, title, cbar_label, save_path, cmap="viridis", fmt=".0f"):
    """Generic annotated heatmap."""
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9 + 2), max(5, n * 0.8 + 2)))

    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("To waypoint", fontsize=12)
    ax.set_ylabel("From waypoint", fontsize=12)
    ax.set_title(title, fontsize=14)

    # annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val == 0 and i == j:
                text = "—"
            else:
                text = f"{val:{fmt}}"
            # choose text color for readability
            thresh = (matrix[matrix > 0].min() + matrix.max()) / 2 if matrix.max() > 0 else 0
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Energy and velocity heatmap visualization")
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Use fixed test waypoint set (includes wind)")
    parser.add_argument("-w", "--wind", type=float, nargs=2,
                        default=[0.0, 0.0], metavar=("WX", "WY"))
    parser.add_argument("-d", "--distribution",
                        choices=["uniform", "clustered", "grid"], default="uniform")
    args = parser.parse_args()

    if args.test:
        config = get_test_sim_config()
    else:
        config = SimulationConfig(
            num_targets=args.num_targets,
            seed=args.seed,
            wind_vector=tuple(args.wind),
            distribution=args.distribution,
        )

    params = get_default_params()
    targets = Targets(
        num_targets=config.num_targets,
        bounds=config.bounds,
        waypoint_set=config.waypoint_set,
        distribution=config.distribution,
        seed=config.seed,
    )
    waypoints = targets.generate_waypoints()

    physics = DronePhysics(params, wind_vector=config.wind_vector)
    optimizer = RoutingOptimizer(physics)
    energy_matrix, optimal_velocities = optimizer.build_energy_matrix(waypoints)

    n = len(waypoints)
    labels = [str(i) for i in range(n)]
    wind = np.array(config.wind_vector)
    wind_speed = float(np.linalg.norm(wind))
    wind_str = f"wind = [{wind[0]:.1f}, {wind[1]:.1f}] m/s" if wind_speed > 0.1 else "no wind"

    # build velocity matrix from dict
    vel_matrix = np.zeros((n, n))
    for (i, j), v in optimal_velocities.items():
        vel_matrix[i, j] = v

    # energy heatmap
    plot_heatmap(
        energy_matrix, labels,
        title=f"Energy Cost Matrix [J] ({wind_str})",
        cbar_label="Energy [J]",
        save_path=PLOTS_DIR / "energy_heatmap.png",
        cmap="YlOrRd",
        fmt=".0f",
    )

    # velocity heatmap
    plot_heatmap(
        vel_matrix, labels,
        title=f"Optimal Velocity Matrix [m/s] ({wind_str})",
        cbar_label="Velocity [m/s]",
        save_path=PLOTS_DIR / "velocity_heatmap.png",
        cmap="viridis",
        fmt=".1f",
    )

    # print asymmetry info
    if wind_speed > 0.1:
        asym = energy_matrix - energy_matrix.T
        max_asym_idx = np.unravel_index(np.argmax(np.abs(asym)), asym.shape)
        print(f"\nMax energy asymmetry: {asym[max_asym_idx]:.0f} J "
              f"between waypoints {max_asym_idx[0]}→{max_asym_idx[1]}")


if __name__ == "__main__":
    main()
