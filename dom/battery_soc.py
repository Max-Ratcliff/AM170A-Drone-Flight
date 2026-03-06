"""
Battery State-of-Charge Visualization

Plots cumulative energy draw as the drone traverses each waypoint along the
optimized route, shown as remaining battery percentage. A horizontal line at
20% marks the "return home" threshold.

Usage:
    python dom/battery_soc.py                      # 5 random waypoints, no wind
    python dom/battery_soc.py --test               # fixed test set with wind
    python dom/battery_soc.py -n 6 -s 42 -w 5.0 -2.0 --battery 250000
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

# DJI Phantom 4: 5870 mAh at 15.2 V ≈ 89,224 J. We round to 90 kJ default.
DEFAULT_BATTERY_J = 90_000


def main():
    parser = argparse.ArgumentParser(
        description="Battery state-of-charge along route")
    parser.add_argument("-n", "--num-targets", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Use fixed test waypoint set (includes wind)")
    parser.add_argument("-w", "--wind", type=float, nargs=2,
                        default=[0.0, 0.0], metavar=("WX", "WY"))
    parser.add_argument("-d", "--distribution",
                        choices=["uniform", "clustered", "grid"], default="uniform")
    parser.add_argument("-m", "--method", choices=["brute", "nn", "held_karp"],
                        default="brute")
    parser.add_argument("--battery", type=float, default=DEFAULT_BATTERY_J,
                        help="Battery capacity in Joules (default: 90000)")
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

    method_map = {"nn": "nearest_neighbor", "held_karp": "held_karp", "brute": "brute"}
    order = optimizer.solve_tsp(energy_matrix, method=method_map[args.method])

    battery_cap = args.battery
    wind = np.array(config.wind_vector)
    wind_speed = float(np.linalg.norm(wind))
    wind_str = f"wind = [{wind[0]:.1f}, {wind[1]:.1f}] m/s" if wind_speed > 0.1 else "no wind"

    # compute per-leg energy and cumulative SOC
    leg_labels = [f"WP {order[0]}"]
    leg_energies = [0.0]  # start at 100%

    for k in range(len(order)):
        i = order[k]
        j = order[(k + 1) % len(order)]
        leg_energies.append(energy_matrix[i, j])
        if (k + 1) < len(order):
            leg_labels.append(f"WP {j}")
        else:
            leg_labels.append(f"WP {order[0]}\n(return)")

    cumulative_energy = np.cumsum(leg_energies)
    soc_pct = (1.0 - cumulative_energy / battery_cap) * 100.0

    total_energy = cumulative_energy[-1]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(max(10, len(leg_labels) * 1.2), 6))

    # color segments: green if above 20%, yellow 10-20%, red below 10%
    x = np.arange(len(leg_labels))
    for k in range(len(x) - 1):
        avg_soc = (soc_pct[k] + soc_pct[k + 1]) / 2
        if avg_soc > 20:
            color = "steelblue"
        elif avg_soc > 10:
            color = "orange"
        else:
            color = "red"
        ax.plot([x[k], x[k + 1]], [soc_pct[k], soc_pct[k + 1]],
                "-o", color=color, linewidth=2.5, markersize=8, zorder=5)

    # annotate each point
    for k in range(len(x)):
        energy_j = leg_energies[k]
        if k == 0:
            label = f"{soc_pct[k]:.0f}%"
        else:
            label = f"{soc_pct[k]:.0f}%\n({energy_j:.0f} J)"
        va = "bottom" if soc_pct[k] > 15 else "top"
        offset = 6 if va == "bottom" else -6
        ax.annotate(label, (x[k], soc_pct[k]),
                    textcoords="offset points", xytext=(0, offset),
                    ha="center", va=va, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # 20% return-home threshold
    ax.axhline(20, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="20% return-home threshold")

    # 0% line
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(leg_labels, fontsize=10)
    ax.set_ylabel("Battery SOC [%]", fontsize=12)
    ax.set_xlabel("Route Progress", fontsize=12)
    ax.set_title(f"Battery State-of-Charge Along Route ({wind_str})\n"
                 f"Battery: {battery_cap/1000:.0f} kJ | Total energy: {total_energy/1000:.1f} kJ "
                 f"({total_energy/battery_cap*100:.0f}% of capacity)",
                 fontsize=13)
    ax.set_ylim(min(soc_pct.min() - 5, -5), 105)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.4, axis="y")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "battery_soc.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    # print summary
    print(f"=== Battery SOC ===")
    print(f"Battery capacity: {battery_cap/1000:.0f} kJ")
    print(f"Total route energy: {total_energy/1000:.1f} kJ ({total_energy/battery_cap*100:.1f}%)")
    print(f"Route: {order}")
    for k in range(len(leg_labels)):
        print(f"  {leg_labels[k]:>15}  SOC={soc_pct[k]:6.1f}%  (leg cost: {leg_energies[k]:.0f} J)")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
