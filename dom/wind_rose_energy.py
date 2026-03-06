"""
Wind Rose Energy Visualization

Sweeps wind direction 0–360° at a fixed wind speed and plots total optimized
route energy as a polar chart. Shows which wind directions are best/worst
for a given route.

Usage:
    python dom/wind_rose_energy.py                  # 5 random waypoints, no wind
    python dom/wind_rose_energy.py --test           # fixed test set
    python dom/wind_rose_energy.py -n 6 -s 42 --wind-speed 5.0
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


def compute_route_energy(waypoints, wind_vector, params, method):
    """Compute total optimized route energy for a given wind vector."""
    physics = DronePhysics(params, wind_vector=tuple(wind_vector))
    optimizer = RoutingOptimizer(physics)
    energy_matrix, _ = optimizer.build_energy_matrix(waypoints)
    order = optimizer.solve_tsp(energy_matrix, method=method)

    total = 0.0
    for k in range(len(order)):
        i = order[k]
        j = order[(k + 1) % len(order)]
        total += energy_matrix[i, j]
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Wind rose energy polar plot")
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Use fixed test waypoint set")
    parser.add_argument("--wind-speed", type=float, default=5.0,
                        help="Fixed wind speed magnitude (m/s)")
    parser.add_argument("-d", "--distribution",
                        choices=["uniform", "clustered", "grid"], default="uniform")
    parser.add_argument("-m", "--method", choices=["brute", "nn", "held_karp"],
                        default="brute")
    parser.add_argument("--resolution", type=int, default=36,
                        help="Number of wind directions to sample")
    args = parser.parse_args()

    if args.test:
        config = get_test_sim_config()
    else:
        config = SimulationConfig(
            num_targets=args.num_targets,
            seed=args.seed,
            wind_vector=(0.0, 0.0),
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

    method_map = {"nn": "nearest_neighbor", "held_karp": "held_karp", "brute": "brute"}
    method = method_map[args.method]
    wind_speed = args.wind_speed

    # sweep wind direction
    angles_deg = np.linspace(0, 360, args.resolution, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    energies = []

    print(f"Sweeping {args.resolution} wind directions at {wind_speed:.1f} m/s ...")
    for deg, rad in zip(angles_deg, angles_rad):
        wv = (wind_speed * np.cos(rad), wind_speed * np.sin(rad))
        e = compute_route_energy(waypoints, wv, params, method)
        energies.append(e)
        print(f"  {deg:6.1f}°  →  {e:.0f} J")

    energies = np.array(energies)

    # close the polar plot by appending the first value
    angles_plot = np.append(angles_rad, angles_rad[0])
    energies_plot = np.append(energies, energies[0])

    # --- polar plot ---
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(angles_plot, energies_plot, "o-", color="steelblue", linewidth=2, markersize=4)
    ax.fill(angles_plot, energies_plot, alpha=0.15, color="steelblue")

    # mark best and worst
    i_best = int(np.argmin(energies))
    i_worst = int(np.argmax(energies))
    ax.plot(angles_rad[i_best], energies[i_best], "v", color="green",
            markersize=12, zorder=10, label=f"Best: {angles_deg[i_best]:.0f}° ({energies[i_best]:.0f} J)")
    ax.plot(angles_rad[i_worst], energies[i_worst], "^", color="red",
            markersize=12, zorder=10, label=f"Worst: {angles_deg[i_worst]:.0f}° ({energies[i_worst]:.0f} J)")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(f"Route Energy vs Wind Direction\n(wind speed = {wind_speed:.1f} m/s)",
                 fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "wind_rose_energy.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
